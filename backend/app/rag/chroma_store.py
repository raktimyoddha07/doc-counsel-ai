"""
Chroma persistent vector store + local BGE embeddings for question-specific retrieval (RAG).
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Set, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    # Package-relative import (canonical new structure).
    from .chains import split_document_pages
    from .embeddings import build_embeddings
except ImportError:  # pragma: no cover
    # Flat-import fallback (legacy execution contexts).
    from chains import split_document_pages  # type: ignore
    from embeddings import build_embeddings  # type: ignore


def sanitize_collection_name(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", raw).strip("._-")
    if not s:
        s = "auditlens_default"
    return s[:63] if len(s) > 63 else s


def collection_name_for(
    user_id: int,
    document_id: Optional[int],
    full_document_context: str,
) -> str:
    if document_id is not None:
        raw = f"auditlens_u{int(user_id)}_doc{int(document_id)}"
    else:
        h = hashlib.sha256(full_document_context.encode("utf-8")).hexdigest()[:28]
        raw = f"auditlens_u{int(user_id)}_h{h}"
    return sanitize_collection_name(raw)


def _embeddings():
    # Local BGE model — see embeddings.py. No API key, content stays on-server.
    return build_embeddings()


def _documents_from_context(
    full_document_context: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs: List[Document] = []
    page_chunks = split_document_pages(full_document_context)
    if not page_chunks:
        for i, chunk in enumerate(splitter.split_text(full_document_context)):
            c = (chunk or "").strip()
            if c:
                docs.append(Document(page_content=c, metadata={"page": 1, "chunk": i}))
        return docs

    for page_num, text in page_chunks:
        text = (text or "").strip()
        if not text:
            continue
        sub = splitter.split_text(text)
        for i, chunk in enumerate(sub):
            c = (chunk or "").strip()
            if not c or c == "(No extractable text found on this page.)":
                continue
            docs.append(Document(page_content=c, metadata={"page": int(page_num), "chunk": i}))
    return docs


def ingest_document(
    *,
    persist_directory: str,
    user_id: int,
    document_id: Optional[int],
    full_document_context: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> str:
    """
    Replace any existing Chroma collection for this user/document and embed chunked pages.
    Returns the collection name used.
    """
    name = collection_name_for(user_id, document_id, full_document_context)
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name)
    except Exception:
        pass

    embeddings = _embeddings()
    docs = _documents_from_context(
        full_document_context,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not docs:
        docs = [
            Document(
                page_content="(No extractable text was indexed.)",
                metadata={"page": 1, "chunk": 0},
            )
        ]

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=name,
        persist_directory=persist_directory,
    )
    return name


def retrieve_context(
    *,
    persist_directory: str,
    collection_name: str,
    question: str,
    k: int = 8,
) -> str:
    embeddings = _embeddings()
    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Hybrid retrieval (Migration 4): dense Chroma + sparse BM25 via RRF.
    # BM25 catches exact clause/section/defined-term strings that dense
    # embeddings blur — critical for legal-document QA. Falls back to dense
    # only if the collection has no storable documents to build BM25 from.
    try:
        from .hybrid_retriever import build_hybrid_retriever

        stored = vs.get(include=["documents", "metadatas"])
        chunks: List[Document] = []
        ids = stored.get("ids") or []
        docs = stored.get("documents") or []
        metas = stored.get("metadatas") or []
        for i in range(len(ids)):
            body = (docs[i] if i < len(docs) else "") or ""
            meta = metas[i] if i < len(metas) else {}
            chunks.append(Document(page_content=body, metadata=meta or {}))

        if chunks:
            dense_retriever = vs.as_retriever(search_kwargs={"k": k})
            ensemble = build_hybrid_retriever(
                dense_retriever=dense_retriever,
                chunks=chunks,
                top_k=k,
            )
            results = ensemble.invoke(question)
        else:
            results = vs.similarity_search(question, k=k)
    except Exception:
        # Any hybrid-path failure must not break chat — degrade to dense only.
        results = vs.similarity_search(question, k=k)

    parts: List[str] = []
    seen: Set[Tuple[int, str]] = set()
    for doc in results:
        page = int(doc.metadata.get("page", 1) or 1)
        body = (doc.page_content or "").strip()
        if not body:
            continue
        key = (page, body[:240])
        if key in seen:
            continue
        seen.add(key)
        parts.append(f"[Page {page}]\n{body}")
    return "\n\n".join(parts).strip()


def delete_document_collection(
    *,
    persist_directory: str,
    user_id: int,
    document_id: Optional[int],
    full_document_context: str,
) -> None:
    name = collection_name_for(user_id, document_id, full_document_context)
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name)
    except Exception:
        pass
