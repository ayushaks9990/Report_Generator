from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
import json
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Try to import user-provided vector DB adapter functions
try:
    from vector_db import query_vectordb, initialize_chromadb  # type: ignore
except Exception:
    query_vectordb = None
    initialize_chromadb = None
    logger.debug("vector_db adapter not found; retrieval functions will return placeholders.")


# -------------------------
# Configuration helpers
# -------------------------
DEFAULT_N_RESULTS = int(os.getenv("RAG_DEFAULT_N_RESULTS", "5"))
DEFAULT_MAX_CONTENT_CHARS = int(os.getenv("RAG_MAX_CONTENT_CHARS", "2000"))
DEFAULT_CONTEXT_MAX_ITEMS = int(os.getenv("RAG_CONTEXT_MAX_ITEMS", "5"))

# Optional collection overrides (useful if your vector DB stores different domains separately)
COLLECTION_OVERRIDES = {
    "sales": os.getenv("VECTOR_DB_SALES_COLLECTION") or os.getenv("VECTOR_DB_COLLECTION") or None,
    "marketing": os.getenv("VECTOR_DB_MARKETING_COLLECTION") or os.getenv("VECTOR_DB_COLLECTION") or None,
    "combined": os.getenv("VECTOR_DB_COMBINED_COLLECTION") or os.getenv("VECTOR_DB_COLLECTION") or None,
    "product": os.getenv("VECTOR_DB_PRODUCT_COLLECTION") or os.getenv("VECTOR_DB_COLLECTION") or None,
    "regional": os.getenv("VECTOR_DB_REGIONAL_COLLECTION") or os.getenv("VECTOR_DB_COLLECTION") or None,
    "custom": os.getenv("VECTOR_DB_CUSTOM_COLLECTION") or os.getenv("VECTOR_DB_COLLECTION") or None,
    "all": os.getenv("VECTOR_DB_COLLECTION") or None,
}


def _coerce_query(query: str, analysis_focus: Optional[str] = None) -> str:
    """Append the user focus to the retrieval query so the vector search is better targeted."""
    query = (query or "").strip()
    analysis_focus = (analysis_focus or "").strip()
    if not analysis_focus:
        return query
    return f"{query}\n\nUser analysis focus:\n{analysis_focus}" if query else f"User analysis focus:\n{analysis_focus}"


def _safe_currency(val: Any) -> str:
    """Format numeric currency safely for inclusion in prompts."""
    try:
        if val is None:
            return "N/A"
        v = float(val)
        if abs(v - int(v)) < 0.001:
            return f"${int(v):,}"
        return f"${v:,.2f}"
    except Exception:
        return str(val)


# -------------------------
# Vector DB compatibility layer
# -------------------------

def _normalize_collection_result(init_result: Any) -> Any:
    """Extract a usable collection object from various initializer return shapes."""
    if init_result is None:
        return None

    # Common patterns:
    # 1) (client, collection)
    # 2) {"client": ..., "collection": ...}
    # 3) collection directly
    if isinstance(init_result, tuple) and len(init_result) >= 2:
        return init_result[1]
    if isinstance(init_result, dict):
        for key in ("collection", "col", "db_collection"):
            if key in init_result:
                return init_result[key]
    return init_result


def _call_query_vectordb(collection: Any, query: str, n_results: int, filter_dict: Optional[Dict[str, Any]] = None) -> Any:
    """Call query_vectordb with the adapter signature it supports."""
    if query_vectordb is None:
        raise RuntimeError("vector_db.query_vectordb is not available")

    last_exc: Optional[Exception] = None
    candidates = [
        # Preferred keyword style
        lambda: query_vectordb(collection, query, n_results=n_results, filter_dict=filter_dict),
        # Alternate keyword names
        lambda: query_vectordb(collection, query, n_results=n_results, where=filter_dict),
        # Positional fallback
        lambda: query_vectordb(collection, query, n_results, filter_dict),
        lambda: query_vectordb(collection, query, n_results),
        lambda: query_vectordb(collection, query),
    ]
    for attempt in candidates:
        try:
            return attempt()
        except TypeError as e:
            last_exc = e
            continue
        except Exception as e:
            # Other exceptions should be surfaced to the caller
            raise e
    raise last_exc or RuntimeError("Unable to call query_vectordb with known signatures")


def _get_collection(collection_name: Optional[str] = None) -> Any:
    """Initialize chromadb and return the most appropriate collection object."""
    if initialize_chromadb is None:
        return None

    init_result = initialize_chromadb()
    collection = _normalize_collection_result(init_result)

    # If the initializer returned a client/dict with multiple collections, try common access patterns.
    if collection_name and isinstance(init_result, dict):
        for key in (collection_name, f"{collection_name}_collection", f"{collection_name}Collection"):
            if key in init_result:
                return init_result[key]

    return collection


# -------------------------
# Retrieval and formatting
# -------------------------

def retrieve_relevant_context(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_type: Optional[str] = None,
    analysis_focus: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve raw results from the vector DB. Returns a dict or None on failure."""
    if initialize_chromadb is None or query_vectordb is None:
        logger.warning("vector_db functions not available; cannot fetch real context.")
        return None

    safe_query = _coerce_query(query, analysis_focus=analysis_focus)
    filter_dict = {"type": filter_type} if filter_type else None

    try:
        collection = _get_collection(collection_name)
        if collection is None:
            logger.warning("No vector DB collection available after initialization.")
            return None
    except Exception as e:
        logger.exception("Failed to initialize vector DB collection: %s", e)
        return None

    try:
        results = _call_query_vectordb(collection, safe_query, n_results=n_results, filter_dict=filter_dict)

        # Normalize common shapes for downstream formatting.
        if results is None:
            return None
        if isinstance(results, dict):
            return results
        if isinstance(results, list):
            # Accept either a list of docs, list of dicts, or list of tuples.
            return {"documents": [results], "metadatas": [[]], "distances": [[]]}

        # Some adapters return a raw string or single object
        return {"documents": [[str(results)]], "metadatas": [[{}]], "distances": [[0.0]]}
    except Exception as e:
        logger.exception("Vector DB query failed: %s", e)
        return None


def _extract_nested(block: Any) -> List[Any]:
    """Handle shapes like [[...]] or [...] and always return a flat list."""
    if block is None:
        return []
    if isinstance(block, list) and block and isinstance(block[0], list):
        return list(block[0])
    if isinstance(block, list):
        return list(block)
    return [block]


def format_retrieval_results(results: Optional[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]]]:
    """Normalize and format raw retrieval results into a list of items or a string message."""
    if not results:
        return "No relevant information found."

    try:
        # If a caller already passed a list of formatted items, accept it.
        if isinstance(results, list):
            if results and isinstance(results[0], dict) and {"rank", "content"}.issubset(results[0].keys()):
                return results
            # Otherwise treat as document list.
            docs = results
            metas = [{} for _ in docs]
            dists = [0.0 for _ in docs]
        else:
            documents_block = results.get("documents")
            metadatas_block = results.get("metadatas")
            distances_block = results.get("distances")

            docs = _extract_nested(documents_block)
            metas = _extract_nested(metadatas_block)
            dists = _extract_nested(distances_block)

            if not docs:
                # Also support alternative keys from some adapters.
                for alt_key in ("data", "items", "results"):
                    if alt_key in results and isinstance(results[alt_key], list):
                        docs = results[alt_key]
                        metas = [{} for _ in docs]
                        dists = [0.0 for _ in docs]
                        break

        if not docs:
            return "No relevant information found."

        formatted: List[Dict[str, Any]] = []
        for i, doc in enumerate(docs):
            if doc is None:
                continue

            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            dist = dists[i] if i < len(dists) else 0.0

            try:
                relevance_score = max(0.0, min(1.0, 1.0 - float(dist)))
            except Exception:
                relevance_score = 1.0

            content = doc if isinstance(doc, str) else str(doc)
            formatted.append(
                {
                    "rank": len(formatted) + 1,
                    "relevance_score": relevance_score,
                    "type": (meta or {}).get("type", "unknown"),
                    "content": content,
                    "metadata": meta or {},
                }
            )

        return formatted if formatted else "No relevant information found."
    except Exception as e:
        logger.exception("Failed to format retrieval results: %s", e)
        return "No relevant information found."


def create_context_string(formatted_context: Union[str, List[Dict[str, Any]]], max_items: int = DEFAULT_CONTEXT_MAX_ITEMS) -> str:
    """Create a single prompt-ready context string for the LLM."""
    if isinstance(formatted_context, str):
        return formatted_context

    parts: List[str] = ["Retrieved relevant information:"]
    count = 0

    for item in formatted_context:
        if count >= max_items:
            break
        if not isinstance(item, dict):
            continue

        count += 1
        rank = item.get("rank", count)
        typ = str(item.get("type", "unknown")).upper()
        score = item.get("relevance_score", 1.0)
        parts.append(f"\n{rank}. [{typ}] (Relevance: {score:.2f})")

        content = str(item.get("content", ""))
        if len(content) > DEFAULT_MAX_CONTENT_CHARS:
            content = content[:DEFAULT_MAX_CONTENT_CHARS] + " ...[truncated]"
        parts.append(f"   {content}")

        meta = item.get("metadata", {}) or {}
        meta_parts: List[str] = []

        item_type = str(item.get("type", "")).lower()
        if item_type == "sales":
            product = meta.get("product") or meta.get("product_name") or "N/A"
            revenue = _safe_currency(meta.get("revenue"))
            region = meta.get("region", "N/A")
            quarter = meta.get("quarter", "N/A")
            meta_parts.extend([
                f"Product: {product}",
                f"Revenue: {revenue}",
                f"Region: {region}",
                f"Quarter: {quarter}",
            ])
        elif item_type == "marketing":
            campaign = meta.get("campaign_name") or meta.get("campaign") or "N/A"
            channel = meta.get("channel", "N/A")
            budget = _safe_currency(meta.get("budget"))
            conversions = meta.get("conversions", "N/A")
            meta_parts.extend([
                f"Campaign: {campaign}",
                f"Channel: {channel}",
                f"Budget: {budget}",
                f"Conversions: {conversions}",
            ])
        else:
            if isinstance(meta, dict):
                if "source" in meta:
                    meta_parts.append(f"Source: {meta['source']}")
                if "id" in meta:
                    meta_parts.append(f"ID: {meta['id']}")
                if "region" in meta:
                    meta_parts.append(f"Region: {meta['region']}")
                if "quarter" in meta:
                    meta_parts.append(f"Quarter: {meta['quarter']}")
                if "product" in meta:
                    meta_parts.append(f"Product: {meta['product']}")
                if "campaign_name" in meta:
                    meta_parts.append(f"Campaign: {meta['campaign_name']}")

        if meta_parts:
            parts.append("   " + " | ".join(meta_parts))

    if count == 0:
        return "No relevant information found."
    return "\n".join(parts)


# -------------------------
# Convenience wrappers used by agent.py / report generators
# -------------------------

def _wrap_retrieval(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    filter_type: Optional[str] = None,
    analysis_focus: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> str:
    results = retrieve_relevant_context(
        query,
        n_results=n_results,
        filter_type=filter_type,
        analysis_focus=analysis_focus,
        collection_name=collection_name,
    )
    formatted = format_retrieval_results(results)
    return create_context_string(formatted)


def retrieve_sales_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type="sales",
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("sales"),
    )


def retrieve_marketing_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type="marketing",
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("marketing"),
    )


def retrieve_combined_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type=None,
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("combined"),
    )


# Extra helpers for report_generator.py / app.py compatibility

def retrieve_product_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type="product",
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("product"),
    )


def retrieve_regional_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type="regional",
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("regional"),
    )


def retrieve_custom_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type="custom",
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("custom"),
    )


def retrieve_all_data(query: str, n_results: int = DEFAULT_N_RESULTS, analysis_focus: Optional[str] = None) -> str:
    return _wrap_retrieval(
        query,
        n_results=n_results,
        filter_type=None,
        analysis_focus=analysis_focus,
        collection_name=COLLECTION_OVERRIDES.get("all"),
    )


# Backwards-compatible aliases that some codebases use
retrieve_context = retrieve_relevant_context
retrieve_relevant_data = retrieve_relevant_context


if __name__ == "__main__":
    print("RAG retrieval smoke test")
    q = "Top performing products in North America"
    ctx = retrieve_combined_data(q, n_results=3, analysis_focus="Focus on enterprise customers.")
    try:
        print("\nContext:\n", ctx)
    except Exception:
        print("\nContext (raw):", json.dumps(ctx, indent=2) if isinstance(ctx, (dict, list)) else str(ctx))
