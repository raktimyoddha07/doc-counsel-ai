"""
Embedding model loader.

This is the single place that owns which embedding model backs the Chroma
vector store. Migration 3 (Gemini → BGE local model) swaps the implementation
here without touching the rest of the RAG pipeline.
"""
from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_embeddings(google_api_key: str, embedding_model: str) -> GoogleGenerativeAIEmbeddings:
    """
    Build the embedding model used for both ingest and retrieval.

    Signature kept stable (google_api_key + embedding_model) so the caller in
    chroma_store does not need to change when the underlying model is swapped
    in a later migration.
    """
    return GoogleGenerativeAIEmbeddings(
        model=embedding_model,
        google_api_key=google_api_key,
    )
