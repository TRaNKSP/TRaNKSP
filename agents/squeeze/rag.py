import os as _os_rag
_os_rag.environ["ANONYMIZED_TELEMETRY"] = "False"
_os_rag.environ["CHROMA_TELEMETRY"] = "False"

"""
TRaNKSP — ChromaDB RAG Layer
Collections: squeeze_news, squeeze_filings, squeeze_lifecycle
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("tranksp.rag")

# Lazy-init ChromaDB client
_chroma_client = None
_embedding_fn = None


# Silence ChromaDB telemetry before import
try:
    import chromadb.telemetry.product.posthog as _chroma_ph
    _chroma_ph.Posthog.capture = lambda *a, **kw: None
except Exception:
    pass
try:
    from chromadb.telemetry import TelemetryClient
    TelemetryClient.capture = lambda *a, **kw: None
except Exception:
    pass


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        persist_path = os.path.join("data", "chromadb")
        os.makedirs(persist_path, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_path)
    return _chroma_client


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        _embedding_fn = DefaultEmbeddingFunction()
    return _embedding_fn


def _get_collection(name: str):
    client = _get_client()
    ef = _get_embedding_fn()
    return client.get_or_create_collection(name=name, embedding_function=ef)


# ── News RAG ──────────────────────────────────────────────────────────────────

def store_news(ticker: str, content: str, source: str = "web", run_id: str = ""):
    """Store news chunk for a ticker in squeeze_news collection."""
    try:
        col = _get_collection("squeeze_news")
        doc_id = f"{ticker}_{run_id}_{datetime.utcnow().timestamp()}"
        col.add(
            documents=[content],
            metadatas=[{"ticker": ticker, "source": source, "run_id": run_id, "ts": datetime.utcnow().isoformat()}],
            ids=[doc_id]
        )
    except Exception as e:
        logger.error(f"RAG store_news error for {ticker}: {e}")


def query_news(ticker: str, query: str, n_results: int = 5) -> List[str]:
    """Retrieve relevant news for a ticker via semantic search."""
    try:
        col = _get_collection("squeeze_news")
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, 10),
            where={"ticker": ticker}
        )
        docs = results.get("documents", [[]])[0]
        return docs if docs else []
    except Exception as e:
        logger.warning(f"RAG query_news error for {ticker}: {e}")
        return []


# ── Filings RAG ───────────────────────────────────────────────────────────────

def store_filing(ticker: str, content: str, filing_type: str = "8-K", run_id: str = ""):
    """Store SEC filing chunk in squeeze_filings collection."""
    try:
        col = _get_collection("squeeze_filings")
        doc_id = f"{ticker}_{filing_type}_{run_id}_{datetime.utcnow().timestamp()}"
        col.add(
            documents=[content],
            metadatas=[{"ticker": ticker, "filing_type": filing_type, "run_id": run_id}],
            ids=[doc_id]
        )
    except Exception as e:
        logger.error(f"RAG store_filing error for {ticker}: {e}")


def query_filings(ticker: str, query: str, n_results: int = 3) -> List[str]:
    """Retrieve relevant filing content for a ticker."""
    try:
        col = _get_collection("squeeze_filings")
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, 5),
            where={"ticker": ticker}
        )
        return results.get("documents", [[]])[0] or []
    except Exception as e:
        logger.warning(f"RAG query_filings error for {ticker}: {e}")
        return []


# ── Lifecycle Memory RAG ──────────────────────────────────────────────────────

def store_lifecycle_snapshot(ticker: str, snapshot_text: str, date: str, status: str):
    """Store daily lifecycle snapshot as embedded document for semantic recall."""
    try:
        col = _get_collection("squeeze_lifecycle")
        doc_id = f"{ticker}_{date}"
        try:
            col.delete(ids=[doc_id])
        except Exception:
            pass
        col.add(
            documents=[snapshot_text],
            metadatas=[{"ticker": ticker, "date": date, "status": status}],
            ids=[doc_id]
        )
    except Exception as e:
        logger.error(f"RAG store_lifecycle error for {ticker}: {e}")


def query_lifecycle_memory(ticker: str, query: str = "squeeze progression and reversal", n_results: int = 5) -> str:
    """Retrieve lifecycle history for bearish thesis generation."""
    try:
        col = _get_collection("squeeze_lifecycle")
        results = col.query(
            query_texts=[query],
            n_results=min(n_results, 10),
            where={"ticker": ticker}
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return "No lifecycle history found."
        return "\n---\n".join(docs)
    except Exception as e:
        logger.warning(f"RAG query_lifecycle error for {ticker}: {e}")
        return "Lifecycle memory unavailable."
