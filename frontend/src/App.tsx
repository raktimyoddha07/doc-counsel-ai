import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Viewer, Worker } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.js?url";
import {
  Box,
  Button,
  Chip,
  CssBaseline,
  Paper,
  Stack,
  TextField,
  ThemeProvider,
  Typography,
  createTheme,
} from "@mui/material";

import { useChat } from "./hooks/useChat";

const CHAT_PANE_DEFAULT = 440;
const CHAT_PANE_MIN = 320;
const CHAT_PANE_MAX = 760;
const PDF_PANE_MIN = 360;
const SPLITTER_WIDTH = 12;
const MIN_CENTER_SPACE = 120;

function uniqueSorted(nums: number[]): number[] {
  return Array.from(new Set(nums)).sort((a, b) => a - b);
}

export default function App() {
  const envApiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
  const apiBaseUrl = envApiBaseUrl ?? "http://localhost:8000";

  const [pdfFileUrl, setPdfFileUrl] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState<number>(0);
  const [viewerPageIndex, setViewerPageIndex] = useState(0);

  const [documentContext, setDocumentContext] = useState<string>("");
  const [currentDocumentId, setCurrentDocumentId] = useState<number | null>(null);
  const [recentDocuments, setRecentDocuments] = useState<
    Array<{ id: number; created_at: string; pdf_filename: string; page_count: number }>
  >([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const userHasResizedRef = useRef(false);

  const [authToken, setAuthToken] = useState<string>(() => localStorage.getItem("auditlens_token") ?? "");
  const [authEmail, setAuthEmail] = useState<string>(() => localStorage.getItem("auditlens_email") ?? "");
  const [authPassword, setAuthPassword] = useState("");
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authError, setAuthError] = useState<string | null>(null);
  const [authLoading, setAuthLoading] = useState(false);

  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const [question, setQuestion] = useState("");

  const { isStreaming, error: chatError, assistantText, sendChat } = useChat({
    apiBaseUrl,
  });

  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>(
    [],
  );
  const [streamingAssistantIndex, setStreamingAssistantIndex] = useState<number | null>(null);
  const [showPdfPane, setShowPdfPane] = useState(true);
  const [chatPaneWidth, setChatPaneWidth] = useState(CHAT_PANE_DEFAULT);
  const [pdfPaneWidth, setPdfPaneWidth] = useState(760);
  const [dragging, setDragging] = useState<"pdf" | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [showPdfSplitterHover, setShowPdfSplitterHover] = useState(false);

  const draggingRef = useRef(dragging);
  useEffect(() => {
    draggingRef.current = dragging;
  }, [dragging]);

  const citationPages = useMemo(() => {
    const pages: number[] = [];
    const re = /\[Page\s+(\d+)\]/g;
    for (const match of assistantText.matchAll(re)) {
      const n = Number(match[1]);
      if (Number.isFinite(n) && n >= 1) pages.push(n);
    }
    return uniqueSorted(pages);
  }, [assistantText]);

  useEffect(() => {
    if (streamingAssistantIndex === null) return;
    setMessages((prev) => {
      if (streamingAssistantIndex < 0 || streamingAssistantIndex >= prev.length) return prev;
      return prev.map((m, idx) =>
        idx === streamingAssistantIndex ? { ...m, content: assistantText } : m,
      );
    });
  }, [assistantText, streamingAssistantIndex]);

  useEffect(() => {
    return () => {
      if (pdfFileUrl) URL.revokeObjectURL(pdfFileUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      if (!rootRef.current) return;
      const rect = rootRef.current.getBoundingClientRect();
      if (!showPdfPane) return;

      // Mutual resize: move divider to change BOTH chat and PDF widths.
      const total = Math.max(0, rect.width - SPLITTER_WIDTH);
      const minChat = CHAT_PANE_MIN;
      const maxChat = Math.min(CHAT_PANE_MAX, total - PDF_PANE_MIN);
      const rawChat = e.clientX - rect.left;
      const nextChat = Math.max(minChat, Math.min(maxChat, rawChat));
      const nextPdf = Math.max(PDF_PANE_MIN, total - nextChat);

      setChatPaneWidth(nextChat);
      setPdfPaneWidth(nextPdf);
    };
    const onUp = () => setDragging(null);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging, showPdfPane]);

  useEffect(() => {
    const onResize = () => {
      if (!rootRef.current) return;
      if (userHasResizedRef.current) return;
      if (draggingRef.current) return;
      if (uploading) return;

      const rect = rootRef.current.getBoundingClientRect();
      const containerWidth = rect.width;

      if (!showPdfPane) return;
      const total = Math.max(0, containerWidth - SPLITTER_WIDTH);
      const minChat = CHAT_PANE_MIN;
      const maxChat = Math.min(CHAT_PANE_MAX, total - PDF_PANE_MIN);
      setChatPaneWidth((prevChat) => {
        const nextChat = Math.max(minChat, Math.min(maxChat, prevChat));
        setPdfPaneWidth(Math.max(PDF_PANE_MIN, total - nextChat));
        return nextChat;
      });
    };

    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [showPdfPane, pdfFileUrl, uploading]);

  async function handleAuthSubmit() {
    setAuthError(null);
    setAuthLoading(true);
    try {
      const endpoint = authMode === "register" ? "/auth/register" : "/auth/login";
      const res = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      });
      const data = await res.json().catch(() => ({} as any));
      if (!res.ok) {
        throw new Error((data as any)?.detail || `HTTP ${res.status}`);
      }
      const token = String((data as any)?.token || "");
      if (!token) throw new Error("No token returned by server.");
      setAuthToken(token);
      localStorage.setItem("auditlens_token", token);
      localStorage.setItem("auditlens_email", authEmail);
      setAuthPassword("");
    } catch (e: any) {
      setAuthError(e?.message ?? String(e));
    } finally {
      setAuthLoading(false);
    }
  }

  function handleLogout() {
    setPdfFileUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setPageCount(0);
    setViewerPageIndex(0);
    setAuthToken("");
    setAuthPassword("");
    setAuthEmail("");
    setDocumentContext("");
    setCurrentDocumentId(null);
    setRecentDocuments([]);
    setDocsLoading(false);
    setMessages([]);
    setStreamingAssistantIndex(null);
    localStorage.removeItem("auditlens_token");
    localStorage.removeItem("auditlens_email");
  }

  async function loadRecentDocuments() {
    if (!authToken) return;
    setDocsLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/documents`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (!res.ok) return;
      const data = (await res.json()) as Array<{
        id: number;
        created_at: string;
        pdf_filename: string;
        page_count: number;
      }>;
      setRecentDocuments(data || []);
    } catch {
      // no-op
    } finally {
      setDocsLoading(false);
    }
  }

  async function loadDocumentChats(documentId: number) {
    if (!authToken) return;
    const res = await fetch(`${apiBaseUrl}/documents/${documentId}/chats`, {
      headers: { Authorization: `Bearer ${authToken}` },
    });
    if (!res.ok) return;
    const chats = (await res.json()) as Array<{
      id: number;
      created_at: string;
      question: string;
      assistant_answer: string;
      citation_pages: number[];
    }>;

    // Show Q/A history for this PDF.
    const next: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (const c of chats) {
      next.push({ role: "user", content: c.question });
      next.push({ role: "assistant", content: c.assistant_answer });
    }
    setMessages(next);
    setCurrentDocumentId(documentId);
    setDocumentContext("");
    setPdfFileUrl(null);
    setViewerPageIndex(0);
    const doc = recentDocuments.find((d) => d.id === documentId);
    if (doc) setPageCount(doc.page_count ?? 0);
    setStreamingAssistantIndex(null);
  }

  useEffect(() => {
    if (!authToken) return;
    loadRecentDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authToken]);

  useEffect(() => {
    if (!authToken) return;
    if (currentDocumentId !== null) return;
    if (!recentDocuments.length) return;
    loadDocumentChats(recentDocuments[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recentDocuments, authToken]);

  async function handleUpload(file: File) {
    setUploadError(null);
    setUploading(true);
    try {
      const objectUrl = URL.createObjectURL(file);
      if (pdfFileUrl) URL.revokeObjectURL(pdfFileUrl);
      setPdfFileUrl(objectUrl);
      setViewerPageIndex(0);

      const form = new FormData();
      form.append("pdf", file);

      const res = await fetch(`${apiBaseUrl}/upload`, {
        method: "POST",
        headers: { Authorization: `Bearer ${authToken}` },
        body: form,
      });

      if (!res.ok) {
        const msg = await res.text().catch(() => "");
        throw new Error(`Upload failed: HTTP ${res.status}: ${msg || res.statusText}`);
      }

      const data = (await res.json()) as {
        document_id?: number;
        page_count: number;
        full_document_context: string;
      };

      setCurrentDocumentId(typeof data.document_id === "number" ? data.document_id : null);
      setPageCount(data.page_count ?? 0);
      setDocumentContext(data.full_document_context ?? "");
      // Refresh the recent-PDF chips for this user.
      await loadRecentDocuments();

      setMessages([
        {
          role: "assistant",
          content: "PDF uploaded. Ask an audit/compliance question and I will cite the exact pages.",
        },
      ]);
      setStreamingAssistantIndex(null);
    } catch (e: any) {
      setUploadError(e?.message ?? String(e));
    } finally {
      setUploading(false);
    }
  }

  async function handleSend() {
    const q = question.trim();
    if (!q) return;
    if (!documentContext.trim() && currentDocumentId === null) return;

    const assistantIdx = messages.length + 1;
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "" },
    ]);
    setStreamingAssistantIndex(assistantIdx);
    setQuestion("");

    try {
      const finalText = await sendChat({
        question: q,
        token: authToken,
        documentContext: documentContext.trim() ? documentContext : undefined,
        documentId: currentDocumentId,
      });
      // Avoid a race where `streamingAssistantIndex` gets cleared
      // before the `assistantText -> messages` effect runs.
      setMessages((prev) =>
        prev.map((m, idx) => (idx === assistantIdx ? { ...m, content: finalText } : m)),
      );
    } catch (e: unknown) {
      setStreamingAssistantIndex(null);
      const errText = e instanceof Error ? e.message : "Chat request failed.";
      setMessages((prev) =>
        prev.map((m, idx) => (idx === assistantIdx ? { ...m, content: errText } : m)),
      );
    }

    setStreamingAssistantIndex(null);
  }

  const onJumpToPage = (page1Based: number) => {
    // Force remount with a target page index (0-based).
    setViewerPageIndex(Math.max(0, page1Based - 1));
  };

  const darkTheme = createTheme({
    palette: {
      mode: "dark",
      background: {
        default: "#020617",
        paper: "#0f172a",
      },
      primary: {
        main: "#38bdf8",
      },
    },
    shape: { borderRadius: 12 },
    typography: {
      fontFamily: '"Inter", "Segoe UI", "Roboto", "Arial", sans-serif',
    },
  });

  const renderMessageWithCitations = (content: string) => {
    const regex = /(\[Page\s+\d+\])/g;
    const segments = content.split(regex);
    return segments.map((part, idx) => {
      const match = /^\[Page\s+(\d+)\]$/.exec(part.trim());
      if (!match) {
        return <Fragment key={`txt-${idx}`}>{part}</Fragment>;
      }
      const page = Number(match[1]);
      return (
        <Box
          key={`cite-${idx}-${page}`}
          onClick={() => onJumpToPage(page)}
          component="button"
          title={`Jump to page ${page}`}
          className="mx-0.5 inline-flex h-3 items-center rounded-full border border-sky-500/40 bg-sky-500/15 px-1 text-[8px] font-semibold leading-none text-sky-300 hover:bg-sky-500/25"
          sx={{ cursor: "pointer" }}
        >
          page{page}
        </Box>
      );
    });
  };

  const formatAssistantMessage = (content: string) => {
    // Model output sometimes collapses line breaks into spaces; restore readable lists.
    let formatted = content.replace(/\r\n/g, "\n");
    formatted = formatted.replace(/[ \t]+\n/g, "\n");
    // Preserve tab characters so TSV-like table content keeps column boundaries.
    // Normalize only repeated spaces (not tabs).
    formatted = formatted.replace(/ {2,}/g, " ");
    formatted = formatted.replace(/\n{3,}/g, "\n\n");

    const rawLines = formatted.split("\n").map((l) => l.replace(/[ \t]+$/g, ""));

    const isEmpty = (s: string) => !s.trim();
    const isOptionLine = (s: string) => /^[A-D][\)\.]\s+/.test(s.trimStart());
    const isBulletLine = (s: string) => /^[-*]\s+/.test(s.trimStart());
    const isNumberedLine = (s: string) => /^\d+[\.\)]\s+/.test(s.trimStart());

    // Line-based formatting:
    // - Only treat bullets/numbering when they start a line (prevents breaking years like "2024.").
    // - Ensure list items have a blank line before them when coming after a paragraph.
    const outLines: string[] = [];
    let prevWasListItem = false;

    for (const line of rawLines) {
      if (isEmpty(line)) {
        if (outLines.length && outLines[outLines.length - 1] !== "") outLines.push("");
        prevWasListItem = false;
        continue;
      }

      const trimmed = line.trim();
      const option = isOptionLine(trimmed);
      const bullet = isBulletLine(trimmed);
      const numbered = isNumberedLine(trimmed);
      const isListItem = option || bullet || numbered;

      if (isListItem) {
        if (
          outLines.length &&
          outLines[outLines.length - 1] !== "" &&
          !prevWasListItem
        ) {
          outLines.push("");
        }
        const leftTrimmed = trimmed.trimStart();
        outLines.push(`  ${leftTrimmed}`);
        prevWasListItem = true;
      } else {
        outLines.push(trimmed);
        prevWasListItem = false;
      }
    }

    // Preserve "Answer:" separation from the next numbered block when missing blank lines.
    const joined = outLines.join("\n").trim();
    const fixed = joined.replace(
      /(Answer:\s*[^\n]*)(\n)(?=\d+[\.\)]\s+)/gim,
      "$1\n\n"
    );
    return fixed;
  };

  return (
    !authToken ? (
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Box className="h-full w-full p-4 flex items-center justify-center">
          <Paper className="w-full max-w-md border border-slate-800 bg-slate-900/80 p-5">
            <Typography variant="h6" sx={{ mb: 1 }}>AuditLens Sign In</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: "block" }}>
              Create/login account to keep PDF sessions and chats.
            </Typography>
            <Stack spacing={1.2}>
              <TextField
                label="Email"
                size="small"
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
              />
              <TextField
                label="Password"
                type="password"
                size="small"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
              />
              {authError ? <Typography variant="caption" color="error.main">{authError}</Typography> : null}
              <Stack direction="row" spacing={1}>
                <Button variant="contained" disabled={authLoading} onClick={handleAuthSubmit}>
                  {authMode === "register" ? "Register" : "Login"}
                </Button>
                <Button
                  variant="outlined"
                  disabled={authLoading}
                  onClick={() => setAuthMode((m) => (m === "login" ? "register" : "login"))}
                >
                  Switch to {authMode === "login" ? "Register" : "Login"}
                </Button>
              </Stack>
            </Stack>
          </Paper>
        </Box>
      </ThemeProvider>
    ) : (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box className="h-full w-full p-4">
        <Box ref={rootRef} sx={{ display: "flex", gap: 0, height: "100%", width: "100%", overflow: "hidden" }}>
          <Paper
            className="flex min-h-0 flex-col border border-slate-800 bg-slate-900/80"
            sx={
              showPdfPane
                ? { width: chatPaneWidth, minWidth: CHAT_PANE_MIN, maxWidth: CHAT_PANE_MAX, flexShrink: 0 }
                : { flex: 1, minWidth: CHAT_PANE_MIN }
            }
          >
            <Box className="border-b border-slate-800 p-4">
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="subtitle1" fontWeight={700}>AuditLens</Typography>
                <Stack direction="row" spacing={1} alignItems="center">
                  {pageCount > 0 ? <Chip size="small" label={`${pageCount} pages`} color="primary" variant="outlined" /> : null}
                  <Chip size="small" label={authEmail || "User"} variant="outlined" />
                  <Chip size="small" label="Logout" onClick={handleLogout} variant="outlined" />
                </Stack>
              </Stack>
              {authToken ? (
                <Box sx={{ mb: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                    Recent PDFs (latest 3)
                  </Typography>
                  {docsLoading ? (
                    <Typography variant="caption" color="text.secondary">
                      Loading...
                    </Typography>
                  ) : null}
                  <Stack direction="row" spacing={0.75} sx={{ flexWrap: "wrap" }}>
                    {recentDocuments.map((d) => {
                      const active = d.id === currentDocumentId;
                      const label = d.pdf_filename ? d.pdf_filename : `Document ${d.id}`;
                      return (
                        <Chip
                          key={d.id}
                          size="small"
                          label={label.length > 18 ? label.slice(0, 18) + "…" : label}
                          color={active ? "primary" : "default"}
                          variant={active ? "filled" : "outlined"}
                          onClick={() => loadDocumentChats(d.id)}
                        />
                      );
                    })}
                    {recentDocuments.length === 0 && !docsLoading ? (
                      <Typography variant="caption" color="text.secondary">
                        Upload a PDF to start a session.
                      </Typography>
                    ) : null}
                  </Stack>
                </Box>
              ) : null}
              {!showPdfPane ? (
                <Stack direction="row" spacing={1} sx={{ mb: 1.5 }}>
                  {!showPdfPane ? (
                    <Chip size="small" label="Open PDF" onClick={() => setShowPdfPane(true)} variant="outlined" />
                  ) : null}
                </Stack>
              ) : null}
              <Stack direction="row" alignItems="center" spacing={1.5}>
                <Button component="label" variant="outlined" size="small" disabled={uploading || isStreaming}>
                  Upload PDF
                  <input
                    type="file"
                    accept="application/pdf"
                    hidden
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleUpload(f);
                    }}
                  />
                </Button>
                <Typography variant="caption" color="text.secondary">
                  {uploading ? "Extracting document..." : "Ready"}
                </Typography>
              </Stack>
              {uploadError ? (
                <Typography variant="caption" color="error.main" sx={{ mt: 1, display: "block" }}>
                  {uploadError}
                </Typography>
              ) : null}
            </Box>

            <Box className="chat-scrollbar min-h-0 flex-1 space-y-3 overflow-auto p-4">
              {messages.map((m, idx) => (
                <Box key={idx} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                  <Box
                    className={
                      m.role === "user"
                        ? "max-w-[88%] rounded-2xl rounded-br-md bg-sky-500 px-3 py-2 text-sm text-slate-950 shadow-sm"
                        : "max-w-[95%] rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900 px-3 py-2 text-sm leading-relaxed text-slate-100 shadow-sm"
                    }
                  >
                    {m.role === "user"
                      ? m.content
                      : (
                        <Box sx={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere", lineHeight: 1.7 }}>
                          {renderMessageWithCitations(formatAssistantMessage(m.content))}
                        </Box>
                      )}
                  </Box>
                </Box>
              ))}
              {isStreaming && <Typography variant="caption" color="text.secondary">Generating answer...</Typography>}
              {chatError ? <Typography variant="caption" color="error.main">{chatError}</Typography> : null}
            </Box>

            {citationPages.length > 0 ? (
              <Box className="border-t border-slate-800 p-4">
                <Typography variant="caption" className="mb-2 block uppercase tracking-wide text-slate-400">
                  Cited pages
                </Typography>
                <Box className="flex flex-wrap gap-2">
                  {citationPages.map((p) => (
                    <Box
                      key={p}
                      component="button"
                      onClick={() => onJumpToPage(p)}
                      className="inline-flex size-8 items-center justify-center rounded-full border border-sky-500/45 bg-sky-500/12 text-xs font-semibold text-sky-300 shadow-sm transition-all hover:scale-105 hover:bg-sky-500/25"
                    >
                      {p}
                    </Box>
                  ))}
                </Box>
              </Box>
            ) : null}

            <Box className="border-t border-slate-800 p-4">
              <Stack direction="row" spacing={1} alignItems="flex-end">
                <TextField
                  fullWidth
                  multiline
                  minRows={2}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder={documentContext.trim() || currentDocumentId !== null ? "Ask about this PDF..." : "Upload a PDF first..."}
                  disabled={(documentContext.trim() === "" && currentDocumentId === null) || isStreaming}
                />
                <Button
                  variant="contained"
                  onClick={handleSend}
                  disabled={(documentContext.trim() === "" && currentDocumentId === null) || isStreaming || !question.trim()}
                >
                  Send
                </Button>
              </Stack>
            </Box>
          </Paper>

          {showPdfPane ? (
            <Box
              onMouseDown={(e) => {
                if (e.button !== 0 && e.button !== 2) return;
                e.preventDefault();
                userHasResizedRef.current = true;
                setDragging("pdf");
              }}
              onContextMenu={(e) => e.preventDefault()}
              onMouseEnter={() => setShowPdfSplitterHover(true)}
              onMouseLeave={() => setShowPdfSplitterHover(false)}
              title="Resize PDF pane"
              sx={{
                width: SPLITTER_WIDTH,
                position: "relative",
                borderRadius: 4,
                cursor: "col-resize",
                backgroundColor: "transparent",
                "&::after": {
                  content: '"↔"',
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  fontSize: 12,
                  color: "#7dd3fc",
                  opacity: showPdfSplitterHover || dragging === "pdf" ? 1 : 0,
                  pointerEvents: "none",
                },
              }}
            />
          ) : null}

          {showPdfPane ? (
          <Paper
            className="min-h-0 overflow-hidden border border-slate-800 bg-slate-900/80 p-4"
            sx={{ width: pdfPaneWidth, minWidth: PDF_PANE_MIN, maxWidth: "100%", flexShrink: 0 }}
          >
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.5, minHeight: 32 }}>
              <Typography variant="subtitle1" fontWeight={700}>Document Preview</Typography>
              <Button size="small" variant="text" onClick={() => setShowPdfPane(false)} title="Close PDF" sx={{ minWidth: 28, px: 0.5 }}>
                ×
              </Button>
            </Stack>
            <Box className="h-[calc(100%-2rem)] min-h-0 overflow-hidden rounded-lg border border-slate-800 bg-slate-950/70 p-1">
              <Worker workerUrl={pdfWorkerUrl}>
                {pdfFileUrl ? (
                  <Viewer
                    key={`${pdfFileUrl}-${viewerPageIndex}`}
                    fileUrl={pdfFileUrl}
                    initialPage={viewerPageIndex}
                    defaultScale={1}
                  />
                ) : (
                  <Box className="flex h-full items-center justify-center text-sm text-slate-400">
                    Upload a PDF to preview it here.
                  </Box>
                )}
              </Worker>
            </Box>
          </Paper>
          ) : null}
        </Box>
      </Box>
    </ThemeProvider>
    )
  );
}

