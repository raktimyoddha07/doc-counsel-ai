import { useCallback, useEffect, useRef, useState } from "react";

type UseVoiceRecorderReturn = {
  /** Whether recording is currently active. */
  recording: boolean;
  /** Whether the browser supports MediaRecorder at all. */
  supported: boolean;
  /** Last user-facing error (permission denied, no mic, etc.). */
  error: string | null;
  /** Start recording. Resolves when recording has begun. Throws/sets error on failure. */
  start: () => Promise<void>;
  /** Stop recording and return the captured audio Blob (audio/webm). */
  stop: () => Promise<Blob | null>;
  /** Cancel an in-progress recording and discard audio. */
  cancel: () => void;
};

/**
 * Thin wrapper around the browser MediaRecorder API for capturing voice input
 * to a Blob (audio/webm). The blob is POSTed to /transcribe by the caller.
 *
 * Feature-detected: `supported` is false on browsers without MediaRecorder
 * (e.g. older Safari), so the UI can hide/disable the mic button gracefully.
 */
export function useVoiceRecorder(): UseVoiceRecorderReturn {
  const supported =
    typeof window !== "undefined" &&
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function" &&
    typeof window.MediaRecorder !== "undefined";

  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const stopResolveRef = useRef<((b: Blob | null) => void) | null>(null);

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop();
      streamRef.current = null;
    }
    mediaRecorderRef.current = null;
    chunksRef.current = [];
  }, []);

  // Ensure tracks are released if the component unmounts mid-recording.
  useEffect(() => {
    return () => {
      cleanupStream();
    };
  }, [cleanupStream]);

  const start = useCallback(async () => {
    if (!supported) {
      setError("Voice input is not supported in this browser.");
      return;
    }
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Prefer webm/opus; fall back to whatever the browser offers.
      let mimeType: string | undefined;
      const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
      for (const c of candidates) {
        if (MediaRecorder.isTypeSupported(c)) {
          mimeType = c;
          break;
        }
      }

      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const type = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type });
        chunksRef.current = [];
        cleanupStream();
        setRecording(false);
        stopResolveRef.current?.(blob);
        stopResolveRef.current = null;
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch (e: any) {
      cleanupStream();
      setRecording(false);
      if (e?.name === "NotAllowedError" || e?.name === "SecurityError") {
        setError("Microphone permission was denied. Allow mic access in your browser settings.");
      } else if (e?.name === "NotFoundError") {
        setError("No microphone device was found.");
      } else {
        setError(e?.message ?? String(e));
      }
    }
  }, [supported, cleanupStream]);

  const stop = useCallback((): Promise<Blob | null> => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      cleanupStream();
      setRecording(false);
      return Promise.resolve(null);
    }
    return new Promise<Blob | null>((resolve) => {
      stopResolveRef.current = resolve;
      try {
        recorder.stop();
      } catch {
        cleanupStream();
        setRecording(false);
        stopResolveRef.current = null;
        resolve(null);
      }
    });
  }, [cleanupStream]);

  const cancel = useCallback(() => {
    stopResolveRef.current = null;
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      try {
        recorder.stop();
      } catch {
        // ignore
      }
    }
    cleanupStream();
    chunksRef.current = [];
    setRecording(false);
  }, [cleanupStream]);

  return { recording, supported, error, start, stop, cancel };
}
