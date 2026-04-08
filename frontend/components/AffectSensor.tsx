"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface Props {
  onMoodDetected: (mood: string) => void;
  enabled: boolean;
}

const INTERVAL_MS = 5000;
const MODELS_URL = "/faceapi-models";

const EXPR_TO_MOOD: Record<string, string> = {
  happy: "happy",
  surprised: "curious",
  neutral: "neutral",
  angry: "stressed",
  fearful: "stressed",
  disgusted: "tired",
  sad: "tired",
};

let modelsLoaded = false;
let modelsPromise: Promise<void> | null = null;

async function loadModels() {
  if (modelsLoaded) {
    return;
  }
  if (!modelsPromise) {
    modelsPromise = (async () => {
      const faceapi = await import("@vladmandic/face-api");
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODELS_URL),
      ]);
      modelsLoaded = true;
    })();
  }
  await modelsPromise;
}

async function detectMoodFromVideo(
  video: HTMLVideoElement,
): Promise<{ mood: string; expression: string; confidence: number } | null> {
  const faceapi = await import("@vladmandic/face-api");

  const result = await faceapi
    .detectSingleFace(
      video,
      new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.3 }),
    )
    .withFaceExpressions();

  if (!result) {
    return null;
  }

  const expressions = result.expressions as unknown as Record<string, number>;
  let topExpression = "neutral";
  let topConfidence = 0;

  for (const [expression, confidence] of Object.entries(expressions)) {
    if (confidence > topConfidence) {
      topConfidence = confidence;
      topExpression = expression;
    }
  }

  return {
    mood: EXPR_TO_MOOD[topExpression] ?? "neutral",
    expression: topExpression,
    confidence: topConfidence,
  };
}

export default function AffectSensor({ onMoodDetected, enabled }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "active" | "error">("idle");
  const [lastMood, setLastMood] = useState<string | null>(null);
  const [lastExpression, setLastExpression] = useState<string | null>(null);
  const [noFace, setNoFace] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  const stopCamera = useCallback(() => {
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStatus("idle");
  }, []);

  const runDetection = useCallback(async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      return;
    }

    setAnalyzing(true);
    setNoFace(false);

    try {
      const result = await detectMoodFromVideo(video);
      if (!result) {
        setNoFace(true);
        return;
      }

      setLastMood(result.mood);
      setLastExpression(`${result.expression} ${Math.round(result.confidence * 100)}%`);
      onMoodDetected(result.mood);
    } catch (error) {
      console.error("face-api detection error", error);
    } finally {
      setAnalyzing(false);
    }
  }, [onMoodDetected]);

  const startCamera = useCallback(async () => {
    setStatus("loading");

    try {
      await loadModels();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStatus("active");
      window.setTimeout(() => {
        void runDetection();
      }, 1000);
      intervalRef.current = window.setInterval(() => {
        void runDetection();
      }, INTERVAL_MS);
    } catch {
      setStatus("error");
    }
  }, [runDetection]);

  useEffect(() => {
    if (enabled) {
      void startCamera();
    } else {
      stopCamera();
      setLastMood(null);
      setLastExpression(null);
      setNoFace(false);
    }

    return () => stopCamera();
  }, [enabled, startCamera, stopCamera]);

  if (status === "idle") {
    return null;
  }

  return (
    <div className="inline-flex items-center gap-2">
      <video ref={videoRef} muted playsInline style={{ display: "none" }} aria-hidden="true" />

      {status === "loading" && (
        <span style={badgeStyle("#818cf8", "rgba(99,102,241,0.12)", "rgba(99,102,241,0.35)")}>
          <Dot color="#818cf8" pulse /> Loading models...
        </span>
      )}

      {status === "active" && (
        <span style={badgeStyle("#4ade80", "rgba(34,197,94,0.12)", "rgba(34,197,94,0.35)")}>
          <Dot color={analyzing ? "#facc15" : noFace ? "#f97316" : "#22c55e"} pulse={analyzing} />
          {analyzing
            ? "Reading..."
            : noFace
              ? "No face"
              : lastMood
                ? `${lastMood} | ${lastExpression}`
                : "Camera on"}
        </span>
      )}

      {status === "error" && (
        <span style={badgeStyle("#f87171", "rgba(239,68,68,0.12)", "rgba(239,68,68,0.35)")}>
          <Dot color="#ef4444" /> Camera denied
        </span>
      )}
    </div>
  );
}

function Dot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span
      style={{
        width: 6,
        height: 6,
        borderRadius: "50%",
        background: color,
        display: "inline-block",
        flexShrink: 0,
        animation: pulse ? "pulse 1s infinite" : undefined,
      }}
    />
  );
}

function badgeStyle(color: string, bg: string, border: string) {
  return {
    display: "inline-flex" as const,
    alignItems: "center" as const,
    gap: 5,
    fontSize: 11,
    padding: "3px 8px",
    borderRadius: 6,
    fontWeight: 600,
    background: bg,
    border: `1px solid ${border}`,
    color,
  };
}
