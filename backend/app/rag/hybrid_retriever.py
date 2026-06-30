"""
Hybrid retrieval: dense (BGE/Chroma) + sparse (BM25) merged via Reciprocal
Rank Fusion using LangChain's EnsembleRetriever.

Migration 4 added this layer on top of the existing Chroma dense retrieval.
For legal documents this is the single highest-ROI accuracy improvement:
semantic search misses exact strings like "Section 4(b)(ii)" or a defined term
("Indemnified Party"), while BM25 catches them precisely. Together they cover
both intent and keyword accuracy.

The BM25 index is built in-memory per retrieval call from the same chunked
Document objects the Chroma collection stores. It does not persist to disk
(Chroma handles persistence); the cost is re-tokenizing the document's chunks,
which is cheap relative to embedding.
"""
from __future__ import annotations

from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

# Default ensemble weights: [dense, sparse]. Slightly favor semantic, but keep
# enough BM25 weight to surface exact clause/section/defined-term hits.
DEFAULT_DENSE_WEIGHT = 0.6
DEFAULT_SPARSE_WEIGHT = 0.4


def build_hybrid_retriever(
    *,
    dense_retriever,
    chunks: List[Document],
    top_k: int = 8,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> EnsembleRetriever:
    """
    Combine a Chroma dense retriever with an in-memory BM25 retriever.

    Args:
        dense_retriever: a LangChain retriever (Chroma `.as_retriever()`).
        chunks: the same chunked Documents stored in the Chroma collection;
            used to build the BM25 index.
        top_k: results each retriever contributes before fusion.
        dense_weight / sparse_weight: RRF-style fusion weights.
    """
    bm25_chunks = [d for d in chunks if (d.page_content or "").strip()]
    bm25_retriever = BM25Retriever.from_documents(bm25_chunks or chunks)
    bm25_retriever.k = top_k

    try:
        dense_retriever.search_kwargs["k"] = top_k
    except Exception:
        pass

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )
