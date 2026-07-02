import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Viewer, Worker } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.js?url";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";

import { useChat } from "./hooks/useChat";
import { useVoiceRecorder } from "./hooks/useVoiceRecorder";

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
  const [currentDocumentId, setCurrentDocumentId] = useState<number | null>(() => {
    const saved = localStorage.getItem("auditlens_current_doc");
    const n = saved ? Number(saved) : NaN;
    return Number.isFinite(n) ? n : null;
  });
  const [recentDocuments, setRecentDocuments] = useState<
    Array<{ id: number; created_at: string; pdf_filename: string; page_count: number; domain?: string }>
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
  // LLM provider picker (Migration 6). Default Ollama (local); user can switch to Gemini.
  const [provider, setProvider] = useState<"ollama" | "gemini">(
    () => (localStorage.getItem("auditlens_provider") as "ollama" | "gemini") || "ollama",
  );

  // Domain picker — auto-detected on upload, can be overridden manually.
  const [activeDomain, setActiveDomain] = useState<string>(
    () => localStorage.getItem("auditlens_domain") || "legal",
  );

  const DOMAIN_LABELS: Record<string, string> = {
    legal: "Legal / Contracts",
    accounting: "Accounting / Finance",
    resume: "Resumes / CVs",
    research: "Research Papers",
    medical: "Medical / Clinical",
    insurance: "Insurance Policies",
    technical: "Technical / Engineering",
    hr: "HR Policies",
    government: "Government / Regulatory",
    patents: "Patents / IP",
  };

  const { isStreaming, error: chatError, assistantText, sendChat } = useChat({
    apiBaseUrl,
  });

  const {
    recording,
    supported: micSupported,
    error: micError,
    start: startRecording,
    stop: stopRecording,
  } = useVoiceRecorder();
  const [transcribing, setTranscribing] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);

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

  const [openMenuDocId, setOpenMenuDocId] = useState<number | null>(null);

  useEffect(() => {
    const handleOutsideClick = () => setOpenMenuDocId(null);
    window.addEventListener("click", handleOutsideClick);
    return () => window.removeEventListener("click", handleOutsideClick);
  }, []);

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
    if (currentDocumentId !== null) {
      localStorage.setItem("auditlens_current_doc", String(currentDocumentId));
    } else {
      localStorage.removeItem("auditlens_current_doc");
    }
  }, [currentDocumentId]);

  useEffect(() => {
    localStorage.setItem("auditlens_provider", provider);
  }, [provider]);

  useEffect(() => {
    localStorage.setItem("auditlens_domain", activeDomain);
  }, [activeDomain]);

  async function handleDomainChange(newDomain: string) {
    setActiveDomain(newDomain);
    if (currentDocumentId !== null && authToken) {
      try {
        await fetch(`${apiBaseUrl}/documents/${currentDocumentId}/domain`, {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${authToken}`,
          },
          body: JSON.stringify({ domain: newDomain }),
        });
      } catch {
        // best-effort — domain is still updated locally
      }
    }
  }

  async function handleDeleteDocument(documentId: number) {
    if (!authToken) return;
    try {
      const res = await fetch(`${apiBaseUrl}/documents/${documentId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (res.ok) {
        if (currentDocumentId === documentId) {
          setCurrentDocumentId(null);
          setPdfFileUrl(null);
          setPageCount(0);
          setDocumentContext("");
          setMessages([
            {
              role: "assistant",
              content: "Session cleared. Upload a PDF or select an existing one to start chatting.",
            },
          ]);
        }
        setRecentDocuments((prev) => prev.filter((d) => d.id !== documentId));
      } else {
        const msg = await res.text().catch(() => "");
        alert(`Delete failed: ${msg}`);
      }
    } catch (e: any) {
      alert(`Error deleting document: ${e?.message ?? String(e)}`);
    }
  }

  useEffect(() => {
    if (!dragging) return;
    const onMove = (e: MouseEvent) => {
      if (!rootRef.current) return;
      const rect = rootRef.current.getBoundingClientRect();
      if (!showPdfPane) return;

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

    const next: Array<{ role: "user" | "assistant"; content: string }> = [];
    for (const c of chats) {
      next.push({ role: "user", content: c.question });
      next.push({ role: "assistant", content: c.assistant_answer });
    }
    setMessages(next);
    setCurrentDocumentId(documentId);
    setDocumentContext("");
    setViewerPageIndex(0);
    const doc = recentDocuments.find((d) => d.id === documentId);
    if (doc) {
      setPageCount(doc.page_count ?? 0);
      if (doc.domain) setActiveDomain(doc.domain);
    }
    setStreamingAssistantIndex(null);

    // Restore the original PDF from per-user object storage so the viewer
    // can render it (fetch-to-blob keeps the Bearer token in the header
    // instead of exposing it in a URL).
    try {
      const pdfRes = await fetch(`${apiBaseUrl}/documents/${documentId}/pdf`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (pdfRes.ok) {
        const blob = await pdfRes.blob();
        if (blob && blob.size > 0) {
          const objectUrl = URL.createObjectURL(blob);
          setPdfFileUrl((prev) => {
            if (prev) URL.revokeObjectURL(prev);
            return objectUrl;
          });
        }
      }
    } catch {
      // no-op: viewer just stays hidden if the stored PDF is unavailable
    }
  }

  const chatsRestoredRef = useRef(false);

  useEffect(() => {
    if (!authToken) return;
    loadRecentDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authToken]);

  useEffect(() => {
    if (!authToken) return;
    if (chatsRestoredRef.current) return;
    if (!recentDocuments.length) return;
    chatsRestoredRef.current = true;
    // Prefer the document the user was last viewing (saved in localStorage);
    // fall back to the most recent upload if it's no longer available.
    const savedId = Number(localStorage.getItem("auditlens_current_doc"));
    const targetId = Number.isFinite(savedId) && recentDocuments.some((d) => d.id === savedId)
      ? savedId
      : recentDocuments[0].id;
    loadDocumentChats(targetId);
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
        domain?: string;
      };

      setCurrentDocumentId(typeof data.document_id === "number" ? data.document_id : null);
      setPageCount(data.page_count ?? 0);
      setDocumentContext(data.full_document_context ?? "");
      if (data.domain) {
        setActiveDomain(data.domain);
      }
      await loadRecentDocuments();

      const detectedLabel = data.domain ? (DOMAIN_LABELS[data.domain] ?? data.domain) : "Legal / Contracts";
      setMessages([
        {
          role: "assistant",
          content: `PDF uploaded. Detected document type: ${detectedLabel}. Ask a question and I will cite the exact pages.`,
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
        provider,
        domain: activeDomain,
      });
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

  // Toggle the mic: start recording, or stop + transcribe and fill the question box.
  async function handleMicToggle() {
    setVoiceError(null);
    if (recording) {
      setTranscribing(true);
      try {
        const blob = await stopRecording();
        if (!blob || blob.size === 0) {
          setVoiceError("No audio captured. Try again.");
          return;
        }
        const form = new FormData();
        form.append("audio", blob, "voice.webm");
        const res = await fetch(`${apiBaseUrl}/transcribe`, {
          method: "POST",
          headers: { Authorization: `Bearer ${authToken}` },
          body: form,
        });
        if (!res.ok) {
          const msg = await res.text().catch(() => "");
          setVoiceError(`Transcription failed (HTTP ${res.status}): ${msg || res.statusText}`);
          return;
        }
        const data = (await res.json()) as { text?: string; error?: string };
        if (data.error) {
          setVoiceError(data.error);
          return;
        }
        const text = (data.text ?? "").trim();
        if (!text) {
          setVoiceError("No speech detected. Try speaking more clearly.");
          return;
        }
        setQuestion((prev) => (prev.trim() ? `${prev} ${text}` : text));
      } catch (e: unknown) {
        setVoiceError(e instanceof Error ? e.message : "Voice input failed.");
      } finally {
        setTranscribing(false);
      }
    } else {
      await startRecording();
    }
  }

  const onJumpToPage = (page1Based: number) => {
    setViewerPageIndex(Math.max(0, page1Based - 1));
  };

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
        <Badge
          key={`cite-${idx}-${page}`}
          onClick={() => onJumpToPage(page)}
          title={`Jump to page ${page}`}
          className="mx-0.5 inline-flex h-3 cursor-pointer items-center rounded-full border border-sky-500/40 bg-sky-500/15 px-1 text-[8px] font-semibold leading-none text-sky-300 hover:bg-sky-500/25"
        >
          page{page}
        </Badge>
      );
    });
  };

  const formatAssistantMessage = (content: string) => {
    let formatted = content.replace(/\r\n/g, "\n");
    formatted = formatted.replace(/[ \t]+\n/g, "\n");
    formatted = formatted.replace(/ {2,}/g, " ");
    formatted = formatted.replace(/\n{3,}/g, "\n\n");

    const rawLines = formatted.split("\n").map((l) => l.replace(/[ \t]+$/g, ""));

    const isEmpty = (s: string) => !s.trim();
    const isOptionLine = (s: string) => /^[A-D][\)\.]\s+/.test(s.trimStart());
    const isBulletLine = (s: string) => /^[-*]\s+/.test(s.trimStart());
    const isNumberedLine = (s: string) => /^\d+[\.\)]\s+/.test(s.trimStart());

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

    const joined = outLines.join("\n").trim();
    const fixed = joined.replace(
      /(Answer:\s*[^\n]*)(\n)(?=\d+[\.\)]\s+)/gim,
      "$1\n\n"
    );
    return fixed;
  };

  if (!authToken) {
    return (
      <div className="flex h-full w-full items-center justify-center p-4">
        <div className="w-full max-w-md rounded-lg border border-slate-800 bg-slate-900/80 p-5">
          <h2 className="mb-1 text-lg font-semibold">AuditLens Sign In</h2>
          <p className="mb-4 text-xs text-muted-foreground">
            Create/login account to keep PDF sessions and chats.
          </p>
          <div className="space-y-3">
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Email</label>
              <Input
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-muted-foreground">Password</label>
              <Input
                type="password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
              />
            </div>
            {authError ? <p className="text-xs text-destructive">{authError}</p> : null}
            <div className="flex gap-2">
              <Button disabled={authLoading} onClick={handleAuthSubmit}>
                {authMode === "register" ? "Register" : "Login"}
              </Button>
              <Button
                variant="outline"
                disabled={authLoading}
                onClick={() => setAuthMode((m) => (m === "login" ? "register" : "login"))}
              >
                Switch to {authMode === "login" ? "Register" : "Login"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const chatPaneStyle = showPdfPane
    ? { width: chatPaneWidth, minWidth: CHAT_PANE_MIN, maxWidth: CHAT_PANE_MAX, flexShrink: 0 }
    : { flex: 1, minWidth: CHAT_PANE_MIN };

  return (
    <div className="h-full w-full p-4">
      <div ref={rootRef} className="flex h-full w-full gap-0 overflow-hidden">
        <div
          className="flex min-h-0 flex-col rounded-lg border border-slate-800 bg-slate-900/80"
          style={chatPaneStyle}
        >
          <div className="border-b border-slate-800 p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-base font-bold">AuditLens</span>
              <div className="flex items-center gap-1.5">
                {pageCount > 0 ? (
                  <Badge variant="outline" className="border-sky-500/40 text-sky-300">
                    {pageCount} pages
                  </Badge>
                ) : null}
                <Badge variant="outline">{authEmail || "User"}</Badge>
                <Badge variant="outline" className="cursor-pointer hover:bg-accent" onClick={handleLogout}>
                  Logout
                </Badge>
              </div>
            </div>
            {authToken ? (
              <div className="mb-2">
                <span className="mb-1 block text-xs text-muted-foreground">
                  Recent PDFs (latest 3)
                </span>
                {docsLoading ? (
                  <span className="text-xs text-muted-foreground">Loading...</span>
                ) : null}
                <div className="flex flex-wrap gap-1.5">
                  {recentDocuments.map((d) => {
                    const active = d.id === currentDocumentId;
                    const label = d.pdf_filename ? d.pdf_filename : `Document ${d.id}`;
                    return (
                      <div key={d.id} className="relative flex items-center">
                        <Badge
                          variant={active ? "default" : "outline"}
                          className="cursor-pointer pr-6 select-none"
                          onClick={() => loadDocumentChats(d.id)}
                        >
                          {label.length > 12 ? label.slice(0, 12) + "…" : label}
                        </Badge>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setOpenMenuDocId(openMenuDocId === d.id ? null : d.id);
                          }}
                          className="absolute right-1 text-slate-400 hover:text-white p-0.5 rounded text-xs leading-none focus:outline-none"
                          title="Options"
                        >
                          ⋮
                        </button>
                        {openMenuDocId === d.id && (
                          <div className="absolute top-full left-0 z-50 mt-1 min-w-[80px] rounded-md border border-slate-800 bg-slate-950 p-1 shadow-lg">
                            <button
                              onClick={async (e) => {
                                e.stopPropagation();
                                setOpenMenuDocId(null);
                                if (window.confirm(`Are you sure you want to delete "${label}"? This will delete the document context and all associated conversations.`)) {
                                  await handleDeleteDocument(d.id);
                                }
                              }}
                              className="w-full text-left rounded px-2 py-1 text-xs text-red-400 hover:bg-slate-900 transition-colors"
                            >
                              Delete
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                  {recentDocuments.length === 0 && !docsLoading ? (
                    <span className="text-xs text-muted-foreground">
                      Upload a PDF to start a session.
                    </span>
                  ) : null}
                </div>
              </div>
            ) : null}
            {!showPdfPane ? (
              <div className="mb-3 flex gap-2">
                <Badge variant="outline" className="cursor-pointer hover:bg-accent" onClick={() => setShowPdfPane(true)}>
                  Open PDF
                </Badge>
              </div>
            ) : null}
            <div className="flex items-center gap-3">
              <Button
                variant="outline"
                size="sm"
                disabled={uploading || isStreaming}
                onClick={() => document.getElementById("pdf-upload-input")?.click()}
              >
                Upload PDF
              </Button>
              <input
                id="pdf-upload-input"
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleUpload(f);
                }}
              />
              <span className="text-xs text-muted-foreground">
                {uploading ? "Extracting & detecting domain..." : "Ready"}
              </span>
              <div className="ml-auto flex items-center gap-2">
                <span className="text-xs text-slate-500">Domain</span>
                <select
                  value={activeDomain}
                  onChange={(e) => handleDomainChange(e.target.value)}
                  disabled={isStreaming || uploading}
                  className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 outline-none transition-colors hover:border-slate-600 focus:border-sky-500 disabled:opacity-50"
                  title="Document type — auto-detected on upload, override if wrong"
                >
                  {Object.entries(DOMAIN_LABELS).map(([id, label]) => (
                    <option key={id} value={id}>{label}</option>
                  ))}
                </select>
              </div>
            </div>
            {uploadError ? (
              <p className="mt-2 block text-xs text-destructive">{uploadError}</p>
            ) : null}
          </div>

          <div className="chat-scrollbar min-h-0 flex-1 space-y-3 overflow-auto p-4">
            {messages.map((m, idx) => (
              <div key={idx} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <div
                  className={
                    m.role === "user"
                      ? "max-w-[88%] rounded-2xl rounded-br-md bg-sky-500 px-3 py-2 text-sm text-slate-950 shadow-sm"
                      : "max-w-[95%] rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900 px-3 py-2 text-sm leading-relaxed text-slate-100 shadow-sm"
                  }
                >
                  {m.role === "user"
                    ? m.content
                    : (
                      <div className="[overflow-wrap:anywhere] [line-height:1.7] whitespace-pre-wrap">
                        {renderMessageWithCitations(formatAssistantMessage(m.content))}
                      </div>
                    )}
                </div>
              </div>
            ))}
            {isStreaming && <span className="text-xs text-muted-foreground">Generating answer...</span>}
            {chatError ? <span className="text-xs text-destructive">{chatError}</span> : null}
          </div>

          {citationPages.length > 0 ? (
            <div className="border-t border-slate-800 p-4">
              <span className="mb-2 block text-xs uppercase tracking-wide text-slate-400">
                Cited pages
              </span>
              <div className="flex flex-wrap gap-2">
                {citationPages.map((p) => (
                  <button
                    key={p}
                    onClick={() => onJumpToPage(p)}
                    className="inline-flex size-8 items-center justify-center rounded-full border border-sky-500/45 bg-sky-500/12 text-xs font-semibold text-sky-300 shadow-sm transition-all hover:scale-105 hover:bg-sky-500/25"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          <div className="border-t border-slate-800 p-4">
            <Textarea
              rows={3}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={documentContext.trim() || currentDocumentId !== null ? "Ask about this PDF..." : "Upload a PDF first..."}
              disabled={(documentContext.trim() === "" && currentDocumentId === null) || isStreaming}
              className="w-full"
            />
            <div className="mt-2 flex items-center gap-2">
              <span className="text-xs text-slate-400">Model</span>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value as "ollama" | "gemini")}
                disabled={isStreaming}
                className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 outline-none transition-colors hover:border-slate-600 focus:border-sky-500 disabled:opacity-50"
                title="Choose the LLM used to answer"
              >
                <option value="ollama">Ollama (Local)</option>
                <option value="gemini">Gemini</option>
              </select>
              {provider === "ollama" ? (
                <span className="text-[11px] text-amber-400/80">running locally, may be slower</span>
              ) : null}

              <div className="ml-auto flex items-center gap-2">
              {micSupported ? (
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={handleMicToggle}
                  disabled={
                    isStreaming ||
                    transcribing ||
                    (documentContext.trim() === "" && currentDocumentId === null)
                  }
                  title={recording ? "Stop recording" : "Speak your question"}
                  className={
                    recording
                      ? "animate-pulse border-red-500 bg-red-500/15 text-red-300 hover:bg-red-500/25"
                      : "border-slate-600 text-slate-300 hover:text-sky-300"
                  }
                >
                  {transcribing ? "…" : recording ? "■" : "🎙"}
                </Button>
              ) : null}
              <Button
                onClick={handleSend}
                disabled={(documentContext.trim() === "" && currentDocumentId === null) || isStreaming || !question.trim()}
              >
                Send
              </Button>
              </div>
            </div>
            {(voiceError || micError) ? (
              <p className="mt-1 block text-xs text-destructive">{voiceError ?? micError}</p>
            ) : null}
            {recording ? (
              <p className="mt-1 block text-xs text-red-300">Recording… click ■ to stop and transcribe.</p>
            ) : null}
          </div>
        </div>

        {showPdfPane ? (
          <div
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
            className="relative cursor-col-resize rounded"
            style={{ width: SPLITTER_WIDTH, backgroundColor: "transparent" }}
          >
            <span
              className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-xs text-sky-300"
              style={{ opacity: showPdfSplitterHover || dragging === "pdf" ? 1 : 0 }}
            >
              ↔
            </span>
          </div>
        ) : null}

        {showPdfPane ? (
          <div
            className="min-h-0 overflow-hidden rounded-lg border border-slate-800 bg-slate-900/80 p-4"
            style={{ width: pdfPaneWidth, minWidth: PDF_PANE_MIN, maxWidth: "100%", flexShrink: 0 }}
          >
            <div className="mb-3 flex min-h-[2rem] items-center justify-between">
              <span className="text-base font-bold">Document Preview</span>
              <Button size="sm" variant="ghost" onClick={() => setShowPdfPane(false)} title="Close PDF">
                ×
              </Button>
            </div>
            <div className="h-[calc(100%-2rem)] min-h-0 overflow-hidden rounded-lg border border-slate-800 bg-slate-950/70 p-1">
              <Worker workerUrl={pdfWorkerUrl}>
                {pdfFileUrl ? (
                  <Viewer
                    key={`${pdfFileUrl}-${viewerPageIndex}`}
                    fileUrl={pdfFileUrl}
                    initialPage={viewerPageIndex}
                    defaultScale={1}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-400">
                    Upload a PDF to preview it here.
                  </div>
                )}
              </Worker>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
