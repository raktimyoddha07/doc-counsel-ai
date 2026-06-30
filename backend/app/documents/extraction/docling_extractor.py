"""
PDF → structured text extraction using Docling (IBM, MIT-licensed).

Replaces the old PyMuPDFLoader-based extraction. Docling uses layout
analysis (DocLayNet) + table structure recognition (TableFormer) for superior
table extraction on legal documents (defined-term tables, obligation schedules,
signature blocks).

Page-level provenance from Docling elements is used to rebuild the same
``[Page N]`` citation format the frontend and LLM chains depend on.
"""
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import HTTPException

from docling.document_converter import DocumentConverter


def _build_full_document_context(
    *,
    elements: List[Dict],
    page_count: int,
    max_full_context_chars: int,
) -> str:
    """Reassemble extracted elements into ``[Page N]`` anchored context."""
    pages: Dict[int, List[str]] = defaultdict(list)
    for el in elements:
        page = el.get("page_number", 1)
        text = (el.get("text") or "").strip()
        if text:
            pages[page].append(text)

    parts: List[str] = []
    for page_num in range(1, page_count + 1):
        texts = pages.get(page_num, [])
        blob = "\n\n".join(texts).strip()
        if not blob:
            blob = "(No extractable text found on this page.)"
        parts.append(f"[Page {page_num}]\n{blob}")

    full_document_context = "\n\n".join(parts).strip()

    if len(full_document_context) > max_full_context_chars:
        raise HTTPException(
            status_code=413,
            detail=f"Extracted context too large ({len(full_document_context)} chars). "
                   f"Please upload a smaller/less dense document.",
        )
    return full_document_context


def extract_pdf_elements(file_path: str) -> List[Dict]:
    """
    Convert a PDF file into element-level dicts with page provenance.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    elements: List[Dict] = []
    for item, _level in result.document.iterate_items():
        page_no = 1
        if item.prov and len(item.prov) > 0:
            page_no = getattr(item.prov[0], "page_no", None) or 1
        text = getattr(item, "text", "") or ""
        if text.strip():
            elements.append({"text": text.strip(), "page_number": page_no})
    return elements


def extract_full_document_context_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    max_pages: int,
    max_full_context_chars: int,
    chunk_size: int = 3000,
    chunk_overlap: int = 100,
) -> Tuple[str, int]:
    """
    Extract structured text from PDF bytes using Docling.

    Returns (full_document_context, page_count) where context uses the
    ``[Page N]`` citation anchor format required by the frontend chips
    and LLM chains.

    ``chunk_size`` and ``chunk_overlap`` are accepted for signature compat
    with the old PyMuPDF path but are unused — Docling produces element-level
    granularity natively. Kept so ``main.py`` call sites don't break.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            tmp_path = tmp.name

        elements = extract_pdf_elements(tmp_path)

        # Determine page count from extracted elements.
        if elements:
            page_count = max(el["page_number"] for el in elements)
        else:
            page_count = 0

        if page_count > max_pages:
            raise HTTPException(
                status_code=413,
                detail=f"PDF page limit exceeded. Max allowed is {max_pages} pages.",
            )

        if page_count == 0:
            page_count = 1  # fallback: single empty page

        full_document_context = _build_full_document_context(
            elements=elements,
            page_count=page_count,
            max_full_context_chars=max_full_context_chars,
        )
        return full_document_context, page_count
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
