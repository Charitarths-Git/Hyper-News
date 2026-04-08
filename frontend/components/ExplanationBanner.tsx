"use client";

import { SurfaceCard } from "@/components/ui/primitives";

interface Props {
  text: string;
  mode: string;
}

const MODE_LABELS: Record<string, string> = {
  rag: "RAG Search",
  rl: "Reinforcement Learning",
  cold_start: "Cold Start",
};

export default function ExplanationBanner({ text, mode }: Props) {
  return (
    <SurfaceCard className="rounded-[26px] bg-[linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0.05))]">
      <div className="space-y-3">
        <div className="flex items-center gap-3">
          <p className="font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent-soft)]">
            AI Reasoning
          </p>
          {mode && (
            <span className="rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 text-xs text-[var(--muted)] backdrop-blur-xl">
              via {MODE_LABELS[mode] ?? mode}
            </span>
          )}
        </div>
        <p className="text-sm leading-7 text-[var(--muted-strong)]">{text}</p>
      </div>
    </SurfaceCard>
  );
}
