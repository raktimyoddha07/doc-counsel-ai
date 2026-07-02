import { useEffect, useRef, useState } from "react";
import { Viewer, Worker } from "@react-pdf-viewer/core";
import { zoomPlugin } from "@react-pdf-viewer/zoom";
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/zoom/lib/styles/index.css";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.js?url";

import { Button } from "@/components/ui/button";

type PdfViewerPanelProps = {
  fileUrl: string | null;
  initialPage?: number;
  jumpToPageSignal?: number; // change this number to force a jump
  onClose?: () => void;
};

export function PdfViewerPanel({
  fileUrl,
  initialPage = 0,
  jumpToPageSignal,
  onClose,
}: PdfViewerPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);

  const zoomPluginInstance = zoomPlugin();
  const { ZoomInButton, ZoomOutButton, ZoomPopover } = zoomPluginInstance;

  // Force the container to have a concrete pixel height — react-pdf-viewer's
  // Viewer needs an explicit, non-percentage height on its parent or the pages
  // collapse to 0px and nothing is visible.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const applyHeight = () => {
      const parent = el.parentElement;
      if (parent) {
        el.style.height = `${parent.clientHeight}px`;
      }
    };
    applyHeight();
    const ro = new ResizeObserver(applyHeight);
    if (el.parentElement) ro.observe(el.parentElement);
    return () => ro.disconnect();
  }, []);

  // React to external jump requests (citation clicks).
  useEffect(() => {
    if (jumpToPageSignal === undefined) return;
    const target = Math.max(1, jumpToPageSignal);
    const el = containerRef.current;
    if (!el) return;
    // react-pdf-viewer renders page elements with a data attribute we can scroll to.
    const pageEl = el.querySelector<HTMLElement>(
      `[data-testid="core__page-layer-${target - 1}"]`,
    );
    if (pageEl) {
      pageEl.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      // Fallback: scroll the inner pages container.
      const inner = el.querySelector(".rpv-core__inner-pages") as HTMLElement | null;
      if (inner) {
        // Estimate by ratio if the exact page element isn't found yet.
        inner.scrollTo({ top: ((target - 1) / Math.max(1, pageCount)) * inner.scrollHeight, behavior: "smooth" });
      }
    }
  }, [jumpToPageSignal, pageCount]);

  return (
    <div className="flex h-full min-h-0 w-full flex-col rounded-lg border border-slate-800 bg-slate-900/80">
      {/* Toolbar */}
      <div className="flex min-h-[2.75rem] flex-shrink-0 items-center justify-between gap-2 border-b border-slate-800 px-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="truncate text-sm font-semibold text-slate-200">Document Preview</span>
          {pageCount > 0 ? (
            <span className="rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-[10px] font-medium text-sky-300">
              {currentPage} / {pageCount}
            </span>
          ) : null}
        </div>
        <div className="flex items-center gap-1">
          {fileUrl ? (
            <>
              <ZoomOutButton />
              <ZoomPopover />
              <ZoomInButton />
              <div className="mx-1 h-5 w-px bg-slate-700" />
            </>
          ) : null}
          {onClose ? (
            <Button size="sm" variant="ghost" onClick={onClose} title="Close PDF" className="h-7 px-2 text-slate-400 hover:text-white">
              ✕
            </Button>
          ) : null}
        </div>
      </div>

      {/* Viewer area: must have explicit, concrete height for react-pdf-viewer */}
      <div className="min-h-0 flex-1 p-2">
        <div ref={containerRef} className="relative h-full w-full overflow-hidden rounded-md border border-slate-800 bg-slate-950/70">
          <Worker workerUrl={pdfWorkerUrl}>
            {fileUrl ? (
              <Viewer
                key={fileUrl}
                fileUrl={fileUrl}
                initialPage={Math.max(0, initialPage)}
                defaultScale={1}
                onDocumentLoad={(e) => {
                  setPageCount(e.doc.numPages);
                }}
                onPageChange={(e) => {
                  setCurrentPage(e.currentPage + 1);
                }}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-400">
                Upload a PDF to preview it here.
              </div>
            )}
          </Worker>
        </div>
      </div>
    </div>
  );
}
