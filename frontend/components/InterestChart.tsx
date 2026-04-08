"use client";

import { useEffect, useMemo, useState } from "react";
import { StateBlock, SurfaceCard } from "@/components/ui/primitives";
import { apiUrl } from "@/lib/api";

interface Props {
  userId: string;
  interests?: Record<string, number>;
}

export default function InterestChart({ userId, interests }: Props) {
  const [fetchedData, setFetchedData] = useState<{ name: string; value: number }[]>([]);

  const derivedData = useMemo(() => {
    if (!interests || Object.keys(interests).length === 0) return [];
    return Object.entries(interests)
      .map(([name, value]) => ({ name, value: Math.max(0, value) }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 8);
  }, [interests]);

  useEffect(() => {
    if (derivedData.length > 0 || !userId) return;
    fetch(apiUrl("/me/profile"))
      .then((r) => r.json())
      .then((profile) => {
        const entries = Object.entries(profile.interests as Record<string, number>)
          .map(([name, value]) => ({ name, value: Math.max(0, value) }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 8);
        setFetchedData(entries);
      })
      .catch(() => {});
  }, [derivedData.length, userId]);

  const data = derivedData.length > 0 ? derivedData : fetchedData;
  const max = Math.max(...data.map((d) => d.value), 0.001);

  if (data.length === 0) {
    return (
      <SurfaceCard>
        <StateBlock
          title="No interests mapped yet"
          description="Read and react to stories to build your interest profile."
        />
      </SurfaceCard>
    );
  }

  return (
    <SurfaceCard className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-[var(--foreground)]">Interest graph</h3>
        <span className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">
          {data.length} categories
        </span>
      </div>
      <div className="space-y-4">
        {data.map((datum) => (
          <div key={datum.name} className="grid gap-2">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="capitalize text-[var(--muted-strong)]">{datum.name}</span>
              <span className="font-mono text-xs text-[var(--accent-soft)]">
                {datum.value.toFixed(2)}
              </span>
            </div>
            <div className="h-3 overflow-hidden rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.03))] backdrop-blur-xl">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),rgba(255,255,255,0.9))] transition-all duration-700"
                style={{ width: `${(datum.value / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </SurfaceCard>
  );
}
