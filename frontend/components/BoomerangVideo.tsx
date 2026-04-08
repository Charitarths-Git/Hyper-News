"use client";

import { useEffect, useRef } from "react";

const VIDEO_STEP = 1 / 30;

export function BoomerangVideo({
  className,
  priority = false,
  mode = "sharp",
}: {
  className?: string;
  priority?: boolean;
  mode?: "sharp" | "blurred";
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRef = useRef<number | null>(null);
  const reversingRef = useRef(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let cancelled = false;

    const cancelFrame = () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };

    const stepReverse = () => {
      if (cancelled || !video) return;
      if (video.currentTime <= VIDEO_STEP) {
        reversingRef.current = false;
        video.currentTime = 0;
        void video.play().catch(() => {});
        return;
      }
      video.currentTime = Math.max(0, video.currentTime - VIDEO_STEP);
      frameRef.current = requestAnimationFrame(stepReverse);
    };

    const startForward = () => {
      cancelFrame();
      reversingRef.current = false;
      video.playbackRate = 1;
      void video.play().catch(() => {});
    };

    const startReverse = () => {
      cancelFrame();
      reversingRef.current = true;
      video.pause();
      frameRef.current = requestAnimationFrame(stepReverse);
    };

    const handleEnded = () => { startReverse(); };
    const handlePlay = () => {
      if (reversingRef.current) { cancelFrame(); reversingRef.current = false; }
    };

    video.addEventListener("ended", handleEnded);
    video.addEventListener("play", handlePlay);
    if (priority) video.load();
    startForward();

    return () => {
      cancelled = true;
      cancelFrame();
      video.removeEventListener("ended", handleEnded);
      video.removeEventListener("play", handlePlay);
    };
  }, [priority]);

  return (
    <video
      ref={videoRef}
      autoPlay
      muted
      playsInline
      preload={priority ? "auto" : "metadata"}
      className={[
        "absolute inset-0 h-full w-full object-cover",
        mode === "blurred" ? "scale-110 blur-[20px] saturate-150 brightness-[0.82]" : "",
        className ?? "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <source
        src="https://res.cloudinary.com/dfonotyfb/video/upload/v1775585556/dds3_1_rqhg7x.mp4"
        type="video/mp4"
      />
    </video>
  );
}
