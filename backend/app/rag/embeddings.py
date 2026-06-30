"""
Embedding model loader.

This is the single place that owns which embedding model backs the Chroma
vector store.

Migration 3 swapped the implementation from Google's hosted
`gemini-embedding-001` to the local `BAAI/bge-large-en-v1.5` model via
`sentence-transformers` / `langchain-huggingface`. This eliminates the Gemini
API dependency for embeddings, removes per-embedding API cost, and keeps all
document content on the server (a hard requirement for the legal use case).

Important: BGE embeddings must be L2-normalized (`normalize_embeddings=True`)
for cosine similarity to behave correctly. Collections built with the previous
Gemini embeddings are incompatible and must be rebuilt once after this change.
"""
from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

# Canonical local model. Centralized so ingest and retrieval never drift.
DEFAULT_BGE_MODEL = "BAAI/bge-large-en-v1.5"


def build_embeddings(
    *,
    model_name: str | None = None,
    device: str | None = None,
):
    """
    Build the embedding model used for both ingest and retrieval.

    The old Gemini signature (google_api_key + embedding_model) is intentionally
    NOT accepted — call sites pass model_name only if overriding the default.
    Keeping the surface narrow prevents accidentally re-introducing a hosted
    dependency that would send document content off-server.

    Args:
        model_name: HuggingFace model id. Defaults to env var
            BGE_EMBEDDING_MODEL, then DEFAULT_BGE_MODEL.
        device: "cpu" or "cuda". Defaults to env var BGE_DEVICE, then "cpu"
            (native Windows dev has no assumed GPU).
    """
    resolved_model = (
        model_name
        or os.getenv("BGE_EMBEDDING_MODEL")
        or DEFAULT_BGE_MODEL
    )
    resolved_device = device or os.getenv("BGE_DEVICE", "cpu")

    return HuggingFaceEmbeddings(
        model_name=resolved_model,
        model_kwargs={"device": resolved_device},
        encode_kwargs={"normalize_embeddings": True},  # required for BGE
    )
