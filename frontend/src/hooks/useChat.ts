import { useCallback, useRef, useState } from "react";

type UseChatParams = {
  apiBaseUrl?: string; // e.g. "http://localhost:8000"
};

type SendChatArgs = {
  question: string;
  token: string;
  documentContext?: string;
  documentId?: number | null;
};

type UseChatReturn = {
  isStreaming: boolean;
  error: string | null;
  assistantText: string;
  sendChat: (args: SendChatArgs) => Promise<string>;
  cancel: () => void;
};

function extractDataPayload(frame: string): string[] {
  // Allow multiple "data:" lines per frame; collect them all.
  const lines = frame.replace(/\r\n/g, "\n").split("\n");
  const dataLines: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    dataLines.push(trimmed.slice("data:".length).trim());
  }
  return dataLines;
}

export function useChat(params: UseChatParams = {}): UseChatReturn {
  const envApiBaseUrl = (import.meta as any).env?.VITE_API_BASE_URL as string | undefined;
  const apiBaseUrl = params.apiBaseUrl ?? envApiBaseUrl ?? "http://localhost:8000";
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [assistantText, setAssistantText] = useState("");

  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const sendChat = useCallback(
    async ({ question, token, documentContext, documentId }: SendChatArgs) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      setIsStreaming(true);
      setError(null);
      setAssistantText("");

      try {
        const res = await fetch(`${apiBaseUrl}/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            question,
            document_context: documentContext ?? "",
            document_id: typeof documentId === "number" ? documentId : null,
          }),
          signal: ac.signal,
        });

        if (!res.ok) {
          const msg = await res.text().catch(() => "");
          throw new Error(`HTTP ${res.status}: ${msg || res.statusText}`);
        }
        if (!res.body) throw new Error("No response body for streaming.");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let finalText = "";
        let doneSeen = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.replace(/\r\n/g, "\n").split("\n\n");
          buffer = parts.pop() ?? "";

          for (const frame of parts) {
            const trimmedFrame = frame.trim();
            if (!trimmedFrame) continue;

            const payloads = extractDataPayload(frame);
            const normalPayloads = payloads.filter((p) => p !== "[DONE]");
            doneSeen = doneSeen || payloads.includes("[DONE]");

            for (const payload of normalPayloads) {
              // Expect: data: {"text":"chunk"}.
              // If parsing fails, ignore the frame.
              try {
                const obj = JSON.parse(payload) as { text?: string };
                if (typeof obj.text === "string" && obj.text.length > 0) {
                  finalText += obj.text;
                  setAssistantText(finalText);
                }
              } catch {
                // Ignore malformed frames.
              }
            }

            if (doneSeen) {
              setAssistantText(finalText);
              setIsStreaming(false);
              return (
                finalText ||
                "No answer text was received. Check backend logs and GEMINI_API_KEY."
              );
            }
          }
        }

        // Parse any leftover partial frame before finishing.
        if (buffer.trim()) {
          const payloads = extractDataPayload(buffer);
          const normalPayloads = payloads.filter((p) => p !== "[DONE]");
          for (const payload of normalPayloads) {
            try {
              const obj = JSON.parse(payload) as { text?: string };
              if (typeof obj.text === "string" && obj.text.length > 0) {
                finalText += obj.text;
              }
            } catch {
              // Ignore malformed frames.
            }
          }
        }

        if (!finalText.trim()) {
          finalText =
            "No answer text was received. Check backend logs and GEMINI_API_KEY.";
        }

        setAssistantText(finalText);
        return finalText;
      } catch (e: any) {
        if (e?.name === "AbortError") {
          setIsStreaming(false);
          return assistantText;
        }
        setError(e?.message ?? String(e));
        setIsStreaming(false);
        throw e;
      } finally {
        setIsStreaming(false);
      }
    },
    [apiBaseUrl, assistantText]
  );

  return { isStreaming, error, assistantText, sendChat, cancel };
}

