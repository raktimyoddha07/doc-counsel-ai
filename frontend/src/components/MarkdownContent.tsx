import { Fragment } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Badge } from "@/components/ui/badge";

/**
 * Renders an assistant message as Markdown, while turning `[Page N]` tokens
 * inside the raw text into clickable citation badges.
 *
 * Citations can appear mid-sentence (e.g. "the indemnity clause [Page 4] states…"),
 * so we strip them out of the markdown source, split the text on them, and
 * interleave rendered markdown blocks with badge chips.
 */

type MarkdownContentProps = {
  content: string;
  onCitationClick?: (page: number) => void;
};

type Segment =
  | { kind: "text"; value: string }
  | { kind: "citation"; page: number };

function splitOnCitations(text: string): Segment[] {
  const regex = /\[Page\s+(\d+)\]/g;
  const segments: Segment[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const page = Number(match[1]);
    if (Number.isFinite(page) && page >= 1) {
      if (match.index > lastIndex) {
        segments.push({ kind: "text", value: text.slice(lastIndex, match.index) });
      }
      segments.push({ kind: "citation", page });
      lastIndex = match.index + match[0].length;
    }
  }
  if (lastIndex < text.length) {
    segments.push({ kind: "text", value: text.slice(lastIndex) });
  }
  return segments;
}

export function MarkdownContent({ content, onCitationClick }: MarkdownContentProps) {
  const segments = splitOnCitations(content ?? "");

  return (
    <div className="markdown-body">
      {segments.map((seg, idx) => {
        if (seg.kind === "citation") {
          return (
            <Badge
              key={`cite-${idx}-${seg.page}`}
              onClick={() => onCitationClick?.(seg.page)}
              title={`Jump to page ${seg.page}`}
              className="mx-0.5 inline-flex h-4 cursor-pointer items-center rounded-full border border-sky-500/40 bg-sky-500/15 px-1.5 align-middle text-[10px] font-semibold leading-none text-sky-300 transition-colors hover:bg-sky-500/30"
            >
              p{seg.page}
            </Badge>
          );
        }
        // Collapse whitespace-only segments to avoid spurious empty paragraphs.
        if (!seg.value.trim()) {
          return <Fragment key={`txt-${idx}`}>{seg.value}</Fragment>;
        }
        return (
          <ReactMarkdown
            key={`md-${idx}`}
            remarkPlugins={[remarkGfm]}
            components={{
              // Render links safely.
              a: ({ node: _node, ...props }) => (
                <a {...props} target="_blank" rel="noopener noreferrer" />
              ),
              table: ({ node: _node, ...props }) => (
                <div className="my-2 overflow-x-auto rounded-md border border-slate-700">
                  <table {...props} />
                </div>
              ),
              th: ({ node: _node, ...props }) => (
                <th
                  {...props}
                  className="border border-slate-700 bg-slate-800/70 px-2.5 py-1.5 text-left text-xs font-semibold"
                />
              ),
              td: ({ node: _node, ...props }) => (
                <td {...props} className="border border-slate-700 px-2.5 py-1.5 text-xs" />
              ),
            }}
          >
            {seg.value}
          </ReactMarkdown>
        );
      })}
    </div>
  );
}
