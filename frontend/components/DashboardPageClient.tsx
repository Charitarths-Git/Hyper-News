"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import InterestChart from "@/components/InterestChart";
import KnowledgeGraphPanel from "@/components/KnowledgeGraphPanel";
import { PageShell } from "@/components/PageShell";
import { ButtonLink, ListPanel, StatTile, SurfaceCard } from "@/components/ui/primitives";
import { apiUrl } from "@/lib/api";
import type { HistoryResponse, Profile, SearchesResponse } from "@/lib/user-types";

interface Props {
  initialProfile: Profile;
}

export default function DashboardPageClient({ initialProfile }: Props) {
  const profile = initialProfile;
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [searches, setSearches] = useState<SearchesResponse | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadDashboard() {
      try {
        const [historyResponse, searchesResponse] = await Promise.all([
          fetch(`${apiUrl("/me/history")}?limit=25`),
          fetch(`${apiUrl("/me/searches")}?limit=25`),
        ]);
        const [historyPayload, searchesPayload] = await Promise.all([
          historyResponse.json(),
          searchesResponse.json(),
        ]);

        if (cancelled) {
          return;
        }

        if (!historyPayload?.error) {
          setHistory(historyPayload as HistoryResponse);
        }
        if (!searchesPayload?.error) {
          setSearches(searchesPayload as SearchesResponse);
        }
      } catch {
        // Keep the initial server-rendered profile visible.
      }
    }

    void loadDashboard();
    return () => {
      cancelled = true;
    };
  }, []);

  const stats = [
    { label: "Articles Read", value: String(profile.articles_read ?? 0) },
    {
      label: "Positive Interactions",
      value: String(profile.total_positive_interactions ?? 0),
    },
    { label: "Current Mood", value: profile.mood || "neutral" },
    {
      label: "Avg Dwell Time",
      value: `${Number(profile.avg_dwell_time || 0).toFixed(1)}s`,
    },
  ];

  return (
    <PageShell
      eyebrow="Dashboard"
      title="Preference intelligence in one place."
      description="Visualise your interest graph, engagement signals, and the profile currently shaping the recommender."
      actions={
        <div className="flex flex-wrap gap-3">
          <ButtonLink href="/feed" variant="ghost">
            Back to feed
          </ButtonLink>
          <ButtonLink href="/profile/settings" variant="secondary">
            Profile settings
          </ButtonLink>
        </div>
      }
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat) => (
          <StatTile key={stat.label} label={stat.label} value={stat.value} />
        ))}
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(320px,0.9fr)]">
        <InterestChart userId={profile.user_id} interests={profile.interests} />
        <SurfaceCard className="space-y-4">
          <h2 className="font-display text-4xl text-[var(--foreground)]">Profile snapshot</h2>
          <div className="space-y-2 text-sm text-[var(--muted)]">
            <p>
              <span className="font-medium text-[var(--muted-strong)]">Top categories:</span>{" "}
              {(profile.top_categories || []).join(", ") || "Not set"}
            </p>
            <p>
              <span className="font-medium text-[var(--muted-strong)]">Interest notes:</span>{" "}
              {profile.interest_text || "Not set"}
            </p>
            <p>
              <span className="font-medium text-[var(--muted-strong)]">Location:</span>{" "}
              {[profile.location_country, profile.location_region].filter(Boolean).join(" / ") ||
                "Not set"}
            </p>
            <p>
              <span className="font-medium text-[var(--muted-strong)]">Bio model:</span>{" "}
              {profile.has_bio_embedding ? "ready" : "text-only"} (
              {profile.bio_embedding_version || "pending"})
            </p>
            <p>
              <span className="font-medium text-[var(--muted-strong)]">Onboarding:</span>{" "}
              {profile.onboarding_completed ? "complete" : "pending"}
            </p>
          </div>
          <Link href="/profile/settings" className="text-sm text-[var(--accent-soft)]">
            Review and adjust profile settings
          </Link>
        </SurfaceCard>
      </div>

      <KnowledgeGraphPanel />

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        <ListPanel
          title="Recent Clicks"
          items={(history?.recent_clicks || profile.recent_clicks.slice(-25)).slice(0, 15)}
        />
        <ListPanel
          title="Recent Negative Signals"
          items={
            (history?.recent_negative_actions || profile.recent_negative_actions.slice(-25)).slice(
              0,
              15,
            )
          }
        />
        <SurfaceCard className="space-y-4">
          <h3 className="text-lg font-semibold text-[var(--foreground)]">Recent Searches</h3>
          <div className="space-y-3 text-sm">
            {(searches?.searches || profile.recent_searches).slice(0, 10).map((entry, index) => (
              <div
                key={`${entry.query_text}-${index}`}
                className="rounded-[18px] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.04))] px-4 py-3"
              >
                <p className="text-[var(--muted-strong)]">{entry.query_text}</p>
                <p className="mt-0.5 text-xs text-[var(--muted)]">{entry.normalized_query}</p>
              </div>
            ))}
          </div>
        </SurfaceCard>
        <SurfaceCard className="space-y-4">
          <h3 className="text-lg font-semibold text-[var(--foreground)]">Recent Feedback</h3>
          <div className="space-y-3 text-sm">
            {(history?.feedback || profile.recent_feedback).slice(0, 10).map((entry, index) => (
              <div
                key={`${entry.article_id}-${index}`}
                className="rounded-[18px] border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.04))] px-4 py-3 text-[var(--muted)]"
              >
                <span className="text-[var(--muted-strong)]">{entry.article_id}</span>
                <span className="mx-2 opacity-40">|</span>
                {entry.action}
              </div>
            ))}
          </div>
        </SurfaceCard>
        <ListPanel title="Recent Entities" items={profile.recent_entities.slice(-15)} />
        <ListPanel title="Recent Sources" items={profile.recent_sources.slice(-15)} />
      </div>
    </PageShell>
  );
}
