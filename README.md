# PdfLens

PdfLens is a document-grounded analysis assistant for PDFs.  
The app extracts structured PDF content (text, tables, images), builds page-aware context, and answers user questions with citations tied to original pages.

This README is intentionally detailed so a new developer can understand the system end-to-end without opening many files first.

### Resume highlights

- **Document-grounded AI assistant** — Built a full-stack app (React/TypeScript, FastAPI, PostgreSQL) that ingests PDFs, preserves page-level context, and answers questions with evidence and clickable page citations.
- **LangChain + Gemini RAG (Chroma)** — PDF chunking, **`GoogleGenerativeAIEmbeddings`**, **Chroma** persistent vector store (`langchain-chroma`), top-k **similarity retrieval** into the LLM prompt for large documents, plus **`langchain-postgres`** session history and SSE streaming.
- **Production-oriented API design** — Implemented authenticated REST (`/auth`, `/upload`, `/documents`, `/chat`), multipart uploads, server-sent events for chat, and optional **asyncpg**/**psycopg** persistence for users, documents, and sessions.

---

## 1. Product Goal

AuditLens is designed to:
- accept a PDF upload
- preserve page context for reliable citation
- return grounded answers tied to document evidence
- provide a practical UI for reading, asking, and validating answers

Key behavior:
- answers should be based on provided PDF content
- citations appear as page tags and are clickable
- UI supports chat + PDF preview + extracted assets side by side

---

## 2. Complete Tech Stack

### Frontend
- **React 18** (TypeScript)
- **Vite**
- **Material UI** (`@mui/material`, Emotion)
- Tailwind-style utility classes in JSX
- **`@react-pdf-viewer/core`** + **`pdfjs-dist`** for in-browser PDF preview

### Backend
- **Python** 3.12+ (3.12–3.13 recommended; some LangChain/Pydantic paths warn on 3.14+ until upstream catches up)
- **FastAPI** + **Uvicorn** (ASGI)
- **LangChain 1.x** — `langchain`, `langchain-core` (via transitive deps), **`langchain-community`** (e.g. `PyMuPDFLoader`), **`langchain-text-splitters`** (`RecursiveCharacterTextSplitter`)
- **`langchain-google-genai`** — `ChatGoogleGenerativeAI` and **`GoogleGenerativeAIEmbeddings`** (Gemini embedding model, default `gemini-embedding-001`)
- **`langchain-chroma`** + **`chromadb`** — on-disk vector index under `backend/chroma_data/` (path configurable); indexes PDF chunks with page metadata for question-specific retrieval
- **`langchain-postgres`** — `PostgresChatMessageHistory` for optional conversation storage
- **PDF processing** — **`pymupdf`** / **`pymupdf4llm`** (helpers and advanced extraction paths), **`pypdf`** where used for validation/metadata
- **`google-generativeai`** — legacy Gemini client still used for some image-caption helpers in `main.py`
- **Persistence (optional)** — **`asyncpg`** connection pool for app tables; **`psycopg`** (async) for LangChain Postgres history
- **`python-multipart`** for uploads, **`python-dotenv`** for configuration

### Integration
- **REST** — auth, upload, document listing, chat payload
- **SSE** (`text/event-stream`) — chat responses framed as `data: ...` plus `[DONE]`
- **PostgreSQL** (optional) — users, documents, chats, LangChain message table when `DATABASE_URL` is set
- **Chroma** (local) — vector retrieval layer when `USE_CHROMA=true` and `GEMINI_API_KEY` is set; does not replace Postgres, complements it

---

## 3. Repository Layout

Top-level directories:
- `backend/` - API logic, extraction pipeline, prompting, streaming
- `frontend/` - application UI and chat client logic
- `description.txt` - original product requirement notes
- `README.md` - this documentation

Backend core files:
- `backend/main.py` — FastAPI app, auth, upload/chat routes, Gemini + SSE helpers
- `backend/retriever.py` — LangChain `PyMuPDFLoader` + `RecursiveCharacterTextSplitter`, builds `[Page N]` context
- `backend/chains.py` — LCEL prompt chain, `ChatGoogleGenerativeAI`, repair pass, table heuristics
- `backend/chroma_rag.py` — Chroma ingest + similarity search; collection naming per user and `document_id` (or content hash if no DB id)
- `backend/database.py` — optional Postgres chat history (`PostgresChatMessageHistory`, session ids)

Frontend core files:
- `frontend/src/App.tsx`
  - main app layout and pane management
  - upload and chat interactions
  - citation rendering and PDF jump behavior
- `frontend/src/hooks/useChat.ts`
  - streaming SSE consumer
  - chunk parsing and incremental assistant text updates
- `frontend/src/styles.css`
  - global dark theme styles and utility CSS
- `frontend/vite.config.ts`
  - build/dev configuration and module alias settings
- `frontend/object-assign.cjs`
  - compatibility shim to resolve `object-assign` in current dependency chain

---

## 4. System Architecture

Runtime processes:

1. **Frontend** — Vite dev server or static production build
2. **Backend** — FastAPI on Uvicorn
3. **PostgreSQL** (optional) — when `DATABASE_URL` is configured: users, uploaded document metadata, chat rows, and LangChain-compatible message history

Data boundary:
- The browser sends **Bearer-authenticated** upload and chat requests (see §6).
- Each `/chat` call may include **`document_context`** from the client and/or **`document_id`** so the server can load stored context from Postgres.
- For **long** documents (above `RAG_FULL_CONTEXT_THRESHOLD` characters), `/chat` pulls **top-k Chroma chunks** into the prompt instead of stuffing the entire PDF, while numeric table **heuristics** still see the **full** stored context when available.
- Questions that look like **whole-document summaries** skip retrieval and still use the full text path.
- Chat replies are streamed over SSE; the UI parses citations like `[Page 3]` and jumps the PDF viewer.

---

## 5. End-to-End Execution Flow

### 5.1 Upload request flow

1. User selects PDF in UI.
2. Frontend creates `FormData` and posts to `POST /upload`.
3. Backend validates:
   - file extension/content type
   - size cap
   - page cap (`MAX_PAGES`)
4. Backend reads the PDF with **LangChain** `PyMuPDFLoader`, splits with **`RecursiveCharacterTextSplitter`**, and assembles **`full_document_context`** with explicit `[Page X]` markers (see `retriever.py`).
5. If Postgres is enabled, the document row is upserted and tied to the authenticated user (recent-docs limit, etc.).
6. When Chroma is enabled, the backend **rebuilds** a per-document vector collection (chunked by page) with **`GoogleGenerativeAIEmbeddings`**.
7. Backend returns JSON including **`page_count`**, **`full_document_context`**, optional **`document_id`**, **`chroma_indexed`**, and **`extracted_assets`**.
8. Frontend stores the payload and enables the question workflow.

### 5.2 Chat request flow

1. User enters question.
2. Frontend sends `POST /chat` with **Bearer token**, **`question`**, optional inline **`document_context`**, and optional **`document_id`** (to load context from Postgres when the preview session uses a stored doc).
3. Backend selects **prompt context**: full `document_context` for short PDFs or “whole document” questions; otherwise **Chroma similarity search** over chunks (with a preamble so the model knows these are retrieved excerpts). Table-related heuristics in **`chains.py`** can still run against the **full** document text.
4. Backend merges system + user prompt via **`chains.py`** (LCEL) and calls **`ChatGoogleGenerativeAI`**.
5. Backend streams the final answer as **SSE** (`data: {"text":"..."}` then `[DONE]`).
6. Frontend parses frames and updates the assistant message incrementally.
7. Frontend extracts `[Page X]` references; citation chips jump the PDF viewer to the cited page.

---

## 6. API Contracts

Protected routes expect header: `Authorization: Bearer <token>` (from `POST /auth/register` or `POST /auth/login`).

### `POST /upload`

Input:
- multipart form-data, field: `pdf`
- `Authorization: Bearer ...`

Output (shape):
```json
{
  "document_id": 12,
  "page_count": 8,
  "full_document_context": "[Page 1] ...",
  "extracted_assets": [],
  "chroma_indexed": true
}
```

`document_id` is null when Postgres is not configured. **`chroma_indexed`** is false if Chroma ingest failed or `USE_CHROMA` is off.

### `POST /chat`

Input:
```json
{
  "question": "What are key risk findings?",
  "document_context": "[Page 1] ...",
  "document_id": 12
}
```

Either non-empty `document_context` or a `document_id` the current user owns is required.

Output:
- content type: `text/event-stream`
- stream emits JSON chunk payloads and `[DONE]`

Other routes: `GET /documents`, `GET /documents/{id}/chats` (when the database and LangChain history table are set up).

---

## 7. Prompting and Response Constraints

Backend prompt logic currently enforces:
- document-grounded responses
- refusal of off-topic requests
- no markdown emphasis spam
- sparse and useful citations
- fact-check behavior:
  - factual statements should be grounded in provided context
  - unsupported claims should be marked as not found in document
- **Enumeration and counting**: questions such as “how many levels (or stages, steps, types, …) are there?” should receive the **count plus a brief description of each item** as given in the PDF, not only the number and one page reference, when the document defines each item.

The system prompt is defined in `backend/chains.py` and mirrored in `backend/main.py` (`build_auditor_system_prompt`) so chat and any legacy helpers stay aligned.

**RAG:** When Chroma retrieval is used, the model is told the `<document_context>` block contains **similarity-ranked excerpts**, not necessarily the entire PDF.

Additional cleanup in stream layer:
- strips star emphasis markers
- reduces repeated identical page tags

---

## 8. Frontend UI Behavior

Layout:
- **Chat** column (left); **PDF preview** column (right, closable)
- draggable vertical splitter between chat and PDF

Pane controls:
- `×` closes pane from top-right of pane header
- closed pane can be reopened from chat header chips
- vertical draggable splitters between panes
- splitter shows tiny two-sided arrow hint and supports drag resizing

Citation behavior:
- in-message page references render as small clickable tags (e.g. `page2`)
- tag click triggers PDF page jump
- optional “Cited pages” button row below the transcript when citations are present

---

## 9. Data Shapes Used in Frontend

Recent-document rows from `/documents`:
- `id`, `created_at`, `pdf_filename`, `page_count`

Chat message state:
- `role: "user" | "assistant"`
- `content: string`

---

## 10. Configuration and Environment

Backend env vars:
- `GEMINI_API_KEY` — required for `/chat` and Chroma embeddings (Gemini via LangChain)
- `GEMINI_MODEL` (default `gemini-1.5-flash`)
- `GEMINI_EMBEDDING_MODEL` (default `gemini-embedding-001`) — used by Chroma ingest + retrieval
- `MAX_PAGES` (default `10`)
- `MAX_FULL_CONTEXT_CHARS` (default `30000`)
- `USE_CHROMA` (default `true`) — set `false` to disable vector retrieval (always full-context prompting)
- `CHROMA_PERSIST_DIRECTORY` — folder for Chroma SQLite index (default `backend/chroma_data`)
- `RAG_TOP_K` (default `8`) — chunks retrieved per question when RAG applies
- `RAG_FULL_CONTEXT_THRESHOLD` (default `14000`) — character count below which the full text is stuffed (no retrieval)
- `DATABASE_URL` — optional Postgres URL for users, documents, chats, LangChain history
- `AUTH_SECRET` — secret for signing auth tokens (change in production)
- `LANGCHAIN_MESSAGE_HISTORY_TABLE` — optional table name override for Postgres chat history

Frontend env var:
- `VITE_API_BASE_URL` (optional; defaults to `http://localhost:8000`)

---

## 11. Error Handling Strategy

Upload-side backend errors:
- invalid type -> `400`
- oversized PDF or page limit -> `413`
- database not configured -> specific auth/upload routes may return `500` where persistence is mandatory

Chat-side:
- LLM or SDK failures are caught and returned as normal SSE text starting with `[Server error]` (HTTP 200) when possible, so the UI stream completes instead of a bare 500
- frontend surfaces errors in the chat panel and message list

Frontend robustness:
- aborts previous stream when sending a new question
- handles malformed SSE frames defensively

---

## 12. Build and Runtime Notes

Known non-fatal build warnings:
- `pdfjs-dist` may emit eval warning in bundle logs
- large chunk size warnings can appear due PDF worker + UI libraries

These warnings do not block normal runtime behavior.

---

## 13. Security and Safety Controls

Current controls:
- sanitize closing tag injection in document context
- block instruction-following from document body (passive data rule)
- vector embeddings stay on the **local Chroma** directory by default (no third-party vector SaaS)

Potential hardening improvements:
- server-side citation validator
- structured claim extraction + per-claim support checks
- stricter policy fallback on unsupported claims

---

## 14. Developer Onboarding Checklist

When joining this project, read in this order:
1. `description.txt`
2. `backend/main.py`
3. `backend/chains.py`
4. `backend/chroma_rag.py`
5. `backend/retriever.py`
6. `frontend/src/App.tsx`
7. `frontend/src/hooks/useChat.ts`
8. `frontend/vite.config.ts`

Then verify:
- upload works
- chat stream works
- citation click jumps page
- pane close/reopen and resize works

---

## 15. Future Improvement Roadmap

Practical next engineering steps:
- add integration tests for upload + streaming chat contract
- add visual regression tests for pane layout
- split large frontend bundle via route/component-level code splitting
- consolidate `build_auditor_system_prompt` into a single module to avoid drift between `main.py` and `chains.py`
- add structured telemetry for extraction/response latency tracking

