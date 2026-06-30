import os
import tempfile
from typing import Dict, List, Tuple

from fastapi import HTTPException
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _coerce_page_num(meta: Dict, fallback: int) -> int:
    # PyMuPDFLoader usually stores page as a zero-based index, but we defensively handle a few variants.
    raw = meta.get("page")
    if raw is None:
        raw = meta.get("page_number")
    if raw is None:
        return fallback
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return fallback
    return page


def _build_full_document_context(
    *,
    chunks: List[Document],
    page_count: int,
    max_full_context_chars: int,
) -> str:
    page_to_texts: Dict[int, List[str]] = {}
    for chunk in chunks:
        meta = chunk.metadata or {}
        page_num = _coerce_page_num(meta, fallback=1)
        page_to_texts.setdefault(page_num, []).append((chunk.page_content or "").strip())

    # Normalize page numbering so that the prompt always uses one-based page anchors: [Page 1], [Page 2], ...
    shift = 1 if page_to_texts and min(page_to_texts.keys()) == 0 else 0

    normalized_page_to_texts: Dict[int, List[str]] = {}
    for raw_page_num, texts in page_to_texts.items():
        normalized_page_to_texts[raw_page_num + shift] = texts

    full_parts: List[str] = []
    for page_num in range(1, page_count + 1):
        texts = normalized_page_to_texts.get(page_num, [])
        # Keep prompt size bounded by allowing missing text (but keep the [Page X] anchor for citations).
        blob = "\n\n".join([t for t in texts if t]).strip()
        if not blob:
            blob = "(No extractable text found on this page.)"
        full_parts.append(f"[Page {page_num}]\n{blob}".strip())

    full_document_context = "\n\n".join(full_parts).strip()

    if len(full_document_context) > max_full_context_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Extracted context too large ({len(full_document_context)} chars). Please upload a smaller/less dense document.",
        )

    return full_document_context


def extract_full_document_context_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    max_pages: int,
    max_full_context_chars: int,
    chunk_size: int = 3000,
    chunk_overlap: int = 100,
) -> Tuple[str, int]:
    """
    Uses LangChain:
    - `PyMuPDFLoader` to extract per-page text
    - `RecursiveCharacterTextSplitter` to split text into context-friendly chunks
    """
    # PyMuPDFLoader expects a file path; we use a temp file.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        page_docs = loader.load()
        page_count = len(page_docs)

        if page_count > max_pages:
            raise HTTPException(
                status_code=413,
                detail=f"PDF page limit exceeded. Max allowed is {max_pages} pages.",
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(page_docs)
        full_document_context = _build_full_document_context(
            chunks=chunks,
            page_count=page_count,
            max_full_context_chars=max_full_context_chars,
        )
        return full_document_context, page_count
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                # Best-effort cleanup.
                pass

