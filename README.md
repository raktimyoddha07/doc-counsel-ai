# Doc Counsel AI

Doc Counsel AI is a full-stack **document-grounded RAG (Retrieval-Augmented Generation)** application. A user uploads a PDF; the backend extracts page-anchored text + tables, indexes them, and then answers the user's questions with streamed answers that include `[Page N]` citations. Clicking a citation jumps the in-browser PDF viewer to that page.

It is positioned for **audit, compliance, and legal document review** — where exact clause retrieval, table cell lookups, and trustworthy page citations are the core trust mechanism.

## 🚀 How to Use

The app has four core sections:

**1. Authentication** — Sign up / log in with email + password. On success you receive a Bearer token that is stored locally and attached to every subsequent request. All document and chat routes require it.

**2. Upload** — Drag or pick a PDF (max ~25 MB, page cap configurable via `MAX_PAGES`). The backend extracts text and tables into a `[Page N]`-anchored context, optionally indexes it into Chroma, and returns the document id. The PDF preview opens in the right-hand viewer pane.

**3. Chat** — Type a question about the document. The answer streams in token-by-token (SSE). Answers are grounded only in the uploaded PDF; off-topic requests are refused. Numeric/table questions ("net profit in 2026") trigger a heuristic table-lookup pass to extract exact cell values.

**4. Citations & PDF Viewer** — Every `[Page N]` in an answer becomes a clickable citation chip. Clicking it scrolls the PDF viewer to that page. A draggable splitter resizes the chat and PDF panes.

A **model selector** lets you switch the LLM provider per-chat between **Ollama** (default, local, privacy-preserving) and **Gemini** (cloud). Your selected document, model, and full chat history are **restored automatically on page reload** — the active document is remembered in `localStorage`, its chat history is fetched from the backend, and the PDF is re-streamed from per-user object storage.

**5. Voice input** — A 🎙 microphone button next to the Send button lets you ask questions by voice. Recording is captured in-browser (MediaRecorder), uploaded to a `/transcribe` endpoint, transcribed on the server with **faster-whisper** (CPU), and the resulting text fills the question box for review before sending.

## ⚙️ Implementation Process

### High-level data flow

```mermaid
flowchart LR
    U[User uploads PDF] --> EX[Docling Extraction]
    EX --> CTX["[Page N] context string"]
    CTX --> DB[(Postgres)]
    CTX --> CH[(Chroma + BGE)]
    U --> STORE[(uploads/ object storage)]

    UQ[User question] --> ROUTE{Routing}
    ROUTE -->|short / whole-doc| FULL[Full context stuffing]
    ROUTE -->|long doc, specific| RAG[Hybrid retrieval]
    CH --> RAG
    RAG --> PROMPT
    FULL --> PROMPT

    MIC[🎤 Mic recording] --> TR[faster-whisper transcribe]
    TR --> BOX[Question box]

    PROMPT[Prompt + Heuristic hints] --> LLM{LLM Provider}
    LLM -->|gemini| G[Gemini API]
    LLM -->|ollama| O[Local Ollama]
    G --> STREAM
    O --> STREAM[SSE text stream]
    STREAM --> UI[Chat UI + citation chips]
```

### 1. PDF Extraction (Docling)

The uploaded PDF bytes are written to a temp file and converted with IBM's **Docling** (`DocumentConverter`), which uses DocLayNet layout analysis and TableFormer for table structure. Each extracted element carries page provenance (`item.prov[0].page_no`), which is used to reassemble the canonical `[Page N]` blocks the rest of the system depends on:

```mermaid
flowchart LR
    PDF[PDF bytes] --> TMP[temp .pdf file]
    TMP --> DOC[DocumentConverter.convert]
    DOC --> E1[Element: text + page_no]
    E1 --> GROUP[Group elements by page]
    GROUP --> OUT["[Page 1] ...\n[Page 2] ..."]
```

### 2. Retrieval (Hybrid: Dense + Sparse)

For long documents, only the most relevant chunks are injected into the prompt. Doc Counsel AI uses **Reciprocal Rank Fusion** via LangChain's `EnsembleRetriever`:

- **Dense** — `BAAI/bge-large-en-v1.5` embeddings (local, L2-normalized) stored in a per-user/per-document **Chroma** collection.
- **Sparse** — in-memory **BM25** index built from the same chunks, excellent at matching exact strings like `Section 4(b)(ii)` or defined terms.

Weights default to `[0.6 dense, 0.4 sparse]`. A routing heuristic decides whether to use full-context stuffing or retrieval:

```mermaid
flowchart LR
    Q[Question] --> OPEN{Open-ended / whole-doc?}
    OPEN -->|yes| FULL[Use full document context]
    OPEN -->|no| SIZE{Context <= threshold?}
    SIZE -->|yes| FULL
    SIZE -->|no| HYBRID[Dense Chroma + BM25 fused via RRF]
    HYBRID --> PROMPT
    FULL --> PROMPT
```

### 3. Answer Generation (LCEL chain + dual provider)

The chain is built with LangChain Expression Language: `prompt_inputs → ChatPromptTemplate → LLM → StrOutputParser`. The LLM is constructed **per request** via a factory that reads the `provider` field sent by the frontend:

- `gemini` → `ChatGoogleGenerativeAI`
- `ollama` → `ChatOllama` (reachability-probed lazily; missing daemon surfaces a clear chat error, never a silent fallback)

A **heuristic table-value hint** tool scans for `metric + year` patterns and injects a candidate numeric value so the model can verify cell lookups. After generation, a **completeness guardrail** detects truncated/dangling answers and triggers a one-shot repair pass.

### 4. Streaming (SSE)

The `/chat` route returns a `StreamingResponse` of `text/event-stream` frames shaped as `data: {"text":"..."}\n\n` terminated by `data: [DONE]`. The frontend `useChat` hook reads the stream with a `ReadableStream` reader, buffers partial frames, and updates the message incrementally. The SSE format is load-bearing — the hook depends on it exactly.

### 5. PDF Object Storage (`uploads/`)

The original uploaded PDF is persisted as per-user object storage so the PDF viewer can be restored after a page reload. The storage root is configured via the `uploads` env var (default `uploads/`, resolved at the project root).

- **Layout:** `uploads/<user_id>/<document_id>__<original_filename>` — the document id guarantees uniqueness on disk while the original filename is preserved for browsing.
- **On upload:** after extraction + Postgres upsert, the raw PDF bytes are written to disk and the path is recorded in the `documents.storage_path` column.
- **On reload:** the frontend calls `GET /documents/{id}/pdf` (Bearer-protected, ownership-enforced), fetches the file as a blob, and feeds it to the PDF viewer — so the user sees exactly the file they uploaded. The on-disk `<id>__` prefix is stripped when serving so the browser sees the original name.

```mermaid
flowchart LR
    UP[/upload PDF/] --> EXTR[Extract + upsert]
    EXTR --> DISK[(uploads/user_id/id__name.pdf)]
    DISK --> PATH[documents.storage_path]
    RELOAD[Page reload] --> GET[GET /documents/id/pdf]
    GET --> DISK
    GET --> BLOB[fetch as blob]
    BLOB --> VIEWER[PDF viewer restored]
```

### 6. Voice Input (faster-whisper)

A microphone button captures audio in-browser via the `MediaRecorder` API (WebM/Opus). The recorded blob is POSTed to `/transcribe`, where the server runs **faster-whisper** (CPU, `base` model) to convert speech to text. faster-whisper handles audio decoding internally through its bundled ffmpeg-based reader, so no external conversion library is needed — the raw webm/opus file is passed directly.

- The WhisperModel is loaded **lazily** on first transcription (the server boots instantly even before the ~145MB model is downloaded).
- Transcription runs in a worker thread (`asyncio.to_thread`) so the event loop isn't blocked.
- The returned text fills (or appends to) the question textarea for user review before sending — it then flows through the normal `/chat` path, inheriting all citation/grounding behavior.
- Model size and device are configurable via `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` env vars.

### 7. Chat Session Persistence

The active document, selected provider, and full chat history survive page reloads:

- **Backend:** chat history is stored in a LangChain Postgres message-history table, scoped per `(user, document)` session. It is fetched on document load via `/documents/{id}/chats` and used as passive context for follow-up questions.
- **Frontend:** `currentDocumentId` and `provider` are persisted to `localStorage`. On boot, the app restores the last-viewed document (falling back to the most recent upload if it's no longer available), reloads its chat history, re-streams its PDF, and re-applies the saved provider selection.

### Key algorithms used

- **RecursiveCharacterTextSplitter** for chunking before embedding (page-aware).
- **Cosine similarity** over normalized BGE vectors (dense retrieval).
- **BM25** term-frequency scoring (sparse retrieval).
- **Reciprocal Rank Fusion** for merging dense + sparse rankings.
- **PBKDF2-HMAC-SHA256** password hashing + **HMAC-signed** stateless Bearer tokens.
- **Regex-based `[Page N]` parsing** for citation extraction on both client and server.

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend framework | React 18 + TypeScript + Vite |
| UI components | shadcn/ui (Radix primitives) + Tailwind CSS |
| PDF viewer | `@react-pdf-viewer/core` + `pdfjs-dist` |
| Backend framework | FastAPI + Uvicorn (Python 3.11+) |
| PDF extraction | Docling (DocLayNet + TableFormer) |
| Embeddings | `BAAI/bge-large-en-v1.5` via `sentence-transformers` |
| Vector store | Chroma (on-disk, per user+document collections) |
| Sparse retrieval | `rank_bm25` + LangChain `EnsembleRetriever` |
| LLM (cloud) | Google Gemini (`langchain-google-genai`) |
| LLM (local, default) | Ollama (`langchain-ollama`) |
| Voice transcription | faster-whisper (`base` model, CPU) |
| PDF storage | Per-user `uploads/` object storage |
| Orchestration | LangChain / LangChain Expression Language (LCEL) |
| Database (optional) | PostgreSQL via `asyncpg` + `langchain-postgres` |
| Streaming | Server-Sent Events (SSE) |
| Auth | HMAC-signed Bearer tokens, PBKDF2 password hashing |
| Containerization | Docker + Docker Compose |

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and npm
- A **Google Gemini API key** (get one from Google AI Studio)
- **PostgreSQL** (optional — only needed for user/document/chat persistence). Without it the app runs statelessly.
- **Ollama** (optional — installed by default as the selected provider; only required if you actually use the local option). Install the native Windows build and run `ollama pull mistral`.
- **ffmpeg** (optional — only needed for the voice/microphone feature). faster-whisper uses it internally to decode the browser's webm/opus audio. A static Windows binary on PATH is sufficient.

### Backend setup

```bash
cd backend
python -m venv venv
# Windows (Git Bash / cmd)
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create `backend/.env` (see `.env details` below), then start the server:

```bash
uvicorn main:app --reload --port 8000
```

The FastAPI app entry point is `backend/main.py`, which imports feature modules from `backend/app/`.

### Frontend setup

```bash
cd frontend
npm install
npm run dev      # starts Vite dev server on http://localhost:5173
```

For a production build:

```bash
npm run build
npm run preview
```

### Other setup (Docker, optional)

A full stack (Postgres + backend + frontend) is defined in `docker-compose.yml`:

```bash
docker compose up --build
```

- Backend is served on port `8000`, frontend on port `3000`, Postgres on `5432`.
- On the primary dev machine (Windows without WSL), native Python/Node is preferred over Docker — see AGENTS.md.

### `.env` details

Create `backend/.env`:

```env
# Required — powers the Gemini LLM (always required, since Gemini is the default provider)
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-1.5-flash

# Required — secret used to sign Bearer auth tokens
AUTH_SECRET=change-this-to-a-long-random-string

# Optional — enables Postgres user/document/chat persistence + LangChain message history
DATABASE_URL=postgresql://postgres:password1234@localhost:5432/PdfLens

# Optional — Ollama (local LLM provider)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Retrieval tuning
USE_CHROMA=true
CHROMA_PERSIST_DIRECTORY=backend/chroma_data
RAG_TOP_K=8
RAG_FULL_CONTEXT_THRESHOLD=14000

# Upload limits
MAX_PAGES=10
MAX_FULL_CONTEXT_CHARS=30000

# PDF object storage folder (relative to project root). Acts as per-user storage.
uploads=uploads

# Voice transcription (faster-whisper, CPU). Optional — only for the mic button.
# First transcription downloads the model (~145MB for "base").
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Gemini LLM access |
| `AUTH_SECRET` | Yes | `change-me-in-env` | Signs Bearer tokens |
| `DATABASE_URL` | No | — | Postgres persistence (off if empty) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Local Ollama daemon |
| `OLLAMA_MODEL` | No | `mistral` | Default Ollama model |
| `USE_CHROMA` | No | `true` | Toggle RAG indexing |
| `RAG_TOP_K` | No | `8` | Chunks retrieved per question |
| `RAG_FULL_CONTEXT_THRESHOLD` | No | `14000` | Chars below which full context is used |
| `MAX_PAGES` | No | `10` | Upload page cap |
| `uploads` | No | `uploads` | Per-user PDF object storage folder |
| `WHISPER_MODEL` | No | `base` | faster-whisper model size |
| `WHISPER_DEVICE` | No | `cpu` | Whisper compute device |
| `WHISPER_COMPUTE_TYPE` | No | `int8` | Whisper compute type |
| `VITE_API_BASE_URL` | No | `http://localhost:8000` | Frontend → backend base URL |

## 🔌 API Endpoints

All protected routes require an `Authorization: Bearer <token>` header.

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | No | Register with `{email, password}` → `{token, user_id, email}` |
| `POST` | `/auth/login` | No | Login with `{email, password}` → `{token, user_id, email}` |
| `POST` | `/upload` | Yes | Multipart PDF upload → `{document_id, page_count, full_document_context, extracted_assets, chroma_indexed}`. Also persists the original PDF to `uploads/<user_id>/`. |
| `POST` | `/chat` | Yes | `{question, document_context|document_id, provider, model?}` → SSE stream of `data: {"text":"..."}` ending in `data: [DONE]` |
| `POST` | `/transcribe` | Yes | Multipart audio (webm/opus) → `{text: "..."}` via faster-whisper (CPU) |
| `GET` | `/documents` | Yes | Lists the user's recent documents (max 3) |
| `GET` | `/documents/{document_id}/pdf` | Yes | Streams the stored original PDF (ownership-enforced) |
| `GET` | `/documents/{document_id}/chats` | Yes | Lists chat history for a document session |

## 📁 Project Structure

```
doc-counsel.ai/
├── backend/
│   ├── main.py                      # FastAPI app, all routes, prompt + table heuristics
│   ├── requirements.txt
│   ├── Dockerfile
│   └── app/
│       ├── rag/
│       │   ├── chains.py            # LCEL chain, system prompt, build_llm() provider factory
│       │   ├── embeddings.py        # BGE local embedding loader
│       │   ├── chroma_store.py      # Chroma ingest + retrieval, page-anchored chunks
│       │   └── hybrid_retriever.py  # BM25 + EnsembleRetriever (RRF fusion)
│       ├── documents/extraction/
│       │   └── docling_extractor.py # PDF → [Page N] context via Docling
│       ├── voice/
│       │   └── transcriber.py       # faster-whisper (CPU) lazy singleton + transcribe
│       ├── chat_history/
│       │   └── database.py          # PostgresChatMessageHistory, session helpers
│       ├── auth/
│       └── core/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts               # Vite + "@" → src/ alias
│   ├── tailwind.config.js
│   └── src/
│       ├── App.tsx                  # Layout, auth, upload, chat, PDF viewer, citations, mic
│       ├── main.tsx
│       ├── hooks/
│       │   ├── useChat.ts           # SSE stream consumer, sends provider per request
│       │   └── useVoiceRecorder.ts  # MediaRecorder wrapper for voice input
│       ├── lib/utils.ts             # cn() Tailwind class merge
│       └── components/ui/           # shadcn/ui primitives (button, input, badge, …)
├── uploads/                         # Per-user PDF object storage (uploads/<user_id>/…)
├── docker-compose.yml
├── AGENTS.md
└── README.md
```

## 📤 Exports

AuditLens does not currently export generated artifacts to a downloadable file. The flows that *produce portable output* are:

- **`/upload` response** returns `full_document_context` (the extracted, page-anchored text) and `extracted_assets` (detected tables) as JSON, so a client can save or post-process them externally.
- **`/documents/{id}/chats`** returns prior Q&A pairs (with parsed `citation_pages`) as JSON, suitable for export to a spreadsheet or report.
- **Chat answers** are streamed as SSE `{"text": "..."}` frames and accumulated by the frontend, so any answer can be copied directly from the UI.

## 📄 License

This project is currently **unlicensed** (no `LICENSE` file is present in the repository). All rights are reserved by the repository owner until a license is added. If you intend to use, fork, or distribute this code, please add an explicit open-source license (e.g. MIT or Apache-2.0) or contact the maintainer.
