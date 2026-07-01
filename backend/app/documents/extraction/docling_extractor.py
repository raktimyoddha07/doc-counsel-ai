"""
PDF → structured text extraction using Docling (IBM, MIT-licensed).

Replaces the old PyMuPDFLoader-based extraction. Docling uses layout
analysis (DocLayNet) + table structure recognition (TableFormer) for superior
table extraction on legal documents (defined-term tables, obligation schedules,
signature blocks).

Page-level provenance from Docling elements is used to rebuild the same
``[Page N]`` citation format the frontend and LLM chains depend on.

Multi-page table stitching
--------------------------
When a table spans a page break, Docling emits two separate ``TableItem``
objects (one per page). Without stitching, the table is split across
``[Page N]`` and ``[Page N+1]``, which breaks the LLM's view of the table,
splits it across Chroma chunks, and produces misleading citations.

``_stitch_adjacent_tables`` detects continuation tables and merges them into a
single logical table cited at its **first** page. Detection is conservative
(adjacent pages only, matching column count) to keep false positives low.
"""
import os
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

from fastapi import HTTPException

from docling.document_converter import DocumentConverter


# ---------------------------------------------------------------------------
# Table detection helpers
# ---------------------------------------------------------------------------

def _is_table(item) -> bool:
    """True if a Docling item is a table (TableItem)."""
    label = getattr(item, "label", None)
    # DocItemLabel.TABLE == "table"; fall back to class name for robustness.
    if label is not None:
        try:
            label = getattr(label, "value", label)
        except Exception:
            pass
        if str(label) == "table":
            return True
    return item.__class__.__name__ == "TableItem"


def _page_number(item) -> int:
    """First-page provenance (1-based) for a Docling item."""
    prov = getattr(item, "prov", None) or []
    if prov:
        page_no = getattr(prov[0], "page_no", None)
        if page_no:
            return int(page_no)
    return 1


def _table_top_y(item) -> float | None:
    """
    Top-edge y-coordinate of the table's first provenance bbox.

    Docling bbox is ``(l, t, r, b)`` in page units where ``t`` is the offset
    from the top edge. A small ``t`` means the table starts near the top of
    the page — a strong signal that it's a continuation of a table from the
    previous page (rather than a new table introduced mid-page).
    """
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None
    bbox = getattr(prov[0], "bbox", None)
    if not bbox:
        return None
    try:
        # bbox may be a sequence or an object with attributes.
        if hasattr(bbox, "__getitem__"):
            return float(bbox[1]) if len(bbox) >= 2 else None
    except (TypeError, ValueError, IndexError):
        return None
    return None


def _table_dims(item) -> Tuple[int, int]:
    """Return (num_rows, num_cols) for a TableItem, (0, 0) if unavailable."""
    data = getattr(item, "data", None)
    if data is None:
        return 0, 0
    num_rows = int(getattr(data, "num_rows", 0) or 0)
    num_cols = int(getattr(data, "num_cols", 0) or 0)
    return num_rows, num_cols


def _table_markdown(item) -> str:
    """Serialize a TableItem to markdown (pipe table). Falls back to .text."""
    try:
        md = item.export_to_markdown()
        if md and md.strip():
            return md.strip()
    except Exception:
        pass
    return (getattr(item, "text", "") or "").strip()


def _table_html(item) -> str:
    """Serialize a TableItem to HTML, if available."""
    try:
        html = item.export_to_html()
        if html and html.strip():
            return html.strip()
    except Exception:
        pass
    return ""


def _table_first_row(item) -> str:
    """
    Best-effort first row text of a table (used to detect a repeated header on
    a continuation page). Returns "" if it can't be determined.
    """
    try:
        md = item.export_to_markdown().strip()
    except Exception:
        return ""
    if not md:
        return ""
    lines = [ln for ln in md.splitlines() if ln.strip().startswith("|")]
    if not lines:
        return ""
    return lines[0].strip()


def _split_header_and_body(markdown: str) -> Tuple[str, str]:
    """
    Split a markdown table into (header_block, body_block).

    The header block is the header row + the ``|---|``` separator that
    follows it. The body block is the remaining data rows. This lets us drop
    a repeated header when stitching two halves of the same table.
    """
    lines = markdown.splitlines()
    pipe_lines = [i for i, ln in enumerate(lines) if ln.strip().startswith("|")]
    if len(pipe_lines) < 2:
        return "", markdown
    header_idx = pipe_lines[0]
    sep_idx = pipe_lines[1]
    # The separator line looks like |---|---|.
    if not all(
        ch in "|-: " for ch in lines[sep_idx].replace("|", "")
    ):
        return "", markdown
    header_block = "\n".join(lines[header_idx:sep_idx + 1])
    body_block = "\n".join(lines[sep_idx + 1:]).strip()
    return header_block, body_block


# ---------------------------------------------------------------------------
# Element extraction
# ---------------------------------------------------------------------------

def extract_pdf_elements(file_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert a PDF file into element-level dicts with page provenance.

    Returns ``(elements, tables)``:

    * ``elements`` — ordered list of ``{text, page_number, kind}`` where
      ``kind`` is ``"text"`` or ``"table"``. Table elements carry their
      serialized markdown in ``text`` plus table metadata
      (``num_rows``, ``num_cols``, ``top_y``).
    * ``tables`` — list of ``{page, num_rows, num_cols, markdown, html}``
      for each (post-stitch) table, surfaced as extracted assets.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    raw_elements: List[Dict] = []

    for item, _level in result.document.iterate_items():
        page_no = _page_number(item)

        if _is_table(item):
            num_rows, num_cols = _table_dims(item)
            md = _table_markdown(item)
            if not md:
                continue
            raw_elements.append({
                "text": md,
                "page_number": page_no,
                "kind": "table",
                "num_rows": num_rows,
                "num_cols": num_cols,
                "top_y": _table_top_y(item),
                "html": _table_html(item),
            })
        else:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                raw_elements.append({
                    "text": text,
                    "page_number": page_no,
                    "kind": "text",
                })

    elements = _stitch_adjacent_tables(raw_elements)

    # Surface final tables as extracted assets (post-stitch).
    tables: List[Dict] = []
    for el in elements:
        if el.get("kind") == "table":
            tables.append({
                "type": "table",
                "page": int(el.get("page_number", 1)),
                "num_rows": int(el.get("num_rows", 0) or 0),
                "num_cols": int(el.get("num_cols", 0) or 0),
                "content": el.get("text", ""),
                "html": el.get("html", ""),
            })

    return elements, tables


# A table starting in the top 15% of the page (by y-offset) is likely a
# continuation of a table from the previous page, not a brand-new table.
_TOP_OF_PAGE_Y_THRESHOLD = 0.15


def _looks_like_continuation(table_b: Dict) -> bool:
    """
    Heuristic: does ``table_b`` (on page P+1) look like a continuation of a
    table that ended on page P? Uses its top-y position when available; falls
    back to ``True`` when position is unknown (column-count + adjacency are
    already strong guards, so we permit the merge by default if we can't tell).
    """
    top_y = table_b.get("top_y")
    if top_y is None:
        return True
    # Docling page coordinates are normalized to [0, 1] in recent versions and
    # to pixel offsets in older ones. A threshold of 0.15 covers normalized
    # coords; for pixel coords it's a permissive "very near the top" test that
    # only triggers for tables genuinely at the top of the page.
    try:
        return float(top_y) <= _TOP_OF_PAGE_Y_THRESHOLD
    except (TypeError, ValueError):
        return True


def _merge_markdown_tables(md_a: str, md_b: str) -> str:
    """
    Concatenate two markdown tables, dropping a repeated header in ``md_b``
    if its first row matches ``md_a``'s header row (common when a continued
    table repeats its column headers on the new page).
    """
    header_a, body_a = _split_header_and_body(md_a)
    header_b, body_b = _split_header_and_body(md_b)

    # If both halves have a recognizable header and the headers match, keep
    # only one header and concatenate the bodies.
    if header_a and header_b:
        first_row_a = header_a.splitlines()[0].strip() if header_a else ""
        first_row_b = header_b.splitlines()[0].strip() if header_b else ""
        if first_row_a and first_row_a == first_row_b:
            return f"{header_a}\n{body_a}\n{body_b}".strip()

    # Otherwise just concatenate raw (each table stays self-describing).
    return f"{md_a}\n{md_b}".strip()


def _stitch_adjacent_tables(elements: List[Dict]) -> List[Dict]:
    """
    Merge tables that continue from page N to page N+1.

    Two adjacent table elements are stitched when:
      1. They are consecutive in document order (no element between them).
      2. Their pages are consecutive (page_B == page_A + 1).
      3. They have the same column count (``num_cols``).
      4. The second table starts near the top of its page (continuation signal),
         or its top position is unknown.

    The merged table keeps the **first** page (citation target) and absorbs the
    second table's rows. Repeated until no more merges occur, so a table that
    spans 3+ pages (N → N+1 → N+2) is fully stitched.
    """
    if not elements:
        return elements

    changed = True
    while changed:
        changed = False
        for i in range(len(elements) - 1):
            a = elements[i]
            b = elements[i + 1]
            if a.get("kind") != "table" or b.get("kind") != "table":
                continue
            page_a = int(a.get("page_number", 0))
            page_b = int(b.get("page_number", 0))
            # For an already-merged table, compare against the LAST page it spans.
            effective_last_page_a = int(a.get("last_page", page_a))
            if page_b != effective_last_page_a + 1:
                continue
            cols_a = int(a.get("num_cols", 0) or 0)
            cols_b = int(b.get("num_cols", 0) or 0)
            # Column count must be known and match. Unknown dims -> don't merge
            # (avoids false positives when Docling couldn't read structure).
            if cols_a == 0 or cols_b == 0 or cols_a != cols_b:
                continue
            if not _looks_like_continuation(b):
                continue

            # Merge b into a.
            merged_rows = int(a.get("num_rows", 0) or 0) + int(b.get("num_rows", 0) or 0)
            merged_md = _merge_markdown_tables(a.get("text", ""), b.get("text", ""))
            merged_html = (a.get("html", "") or "") + (b.get("html", "") or "")

            a["text"] = merged_md
            a["html"] = merged_html
            a["num_rows"] = merged_rows
            # Track the last page this (possibly already-merged) table spans,
            # so a 3+ page span can chain: page 3 absorbs 4, then absorbs 5.
            a["last_page"] = int(b.get("page_number", page_b))
            # num_cols unchanged (they match). page_number stays the first page
            # (citation target).

            del elements[i + 1]
            changed = True
            break  # restart the scan after mutation

    return elements


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

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


def extract_full_document_context_from_pdf_bytes(
    *,
    pdf_bytes: bytes,
    max_pages: int,
    max_full_context_chars: int,
    chunk_size: int = 3000,
    chunk_overlap: int = 100,
) -> Tuple[str, int, List[Dict]]:
    """
    Extract structured text from PDF bytes using Docling.

    Returns ``(full_document_context, page_count, tables)`` where context uses
    the ``[Page N]`` citation anchor format required by the frontend chips and
    LLM chains, and ``tables`` is the list of (stitched) table assets to store
    for downstream legal features.

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

        elements, tables = extract_pdf_elements(tmp_path)

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
        return full_document_context, page_count, tables
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
