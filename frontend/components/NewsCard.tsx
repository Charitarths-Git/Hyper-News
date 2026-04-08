"use client";

import { Pill, SurfaceCard } from "@/components/ui/primitives";

interface Article {
  news_id: string;
  title: string;
  abstract: string;
  category: string;
  source?: string;
  score?: number;
  candidate_source?: string;
  reasons?: string[];
  matched_entities?: string[];
}

interface Props {
  article: Article;
  feedbackState: string | null;
  onFeedback: (action: string) => void;
}

const CATEGORY_LABELS: Record<string, string> = {
  technology: "Tech", sports: "Sports", politics: "Politics",
  entertainment: "Culture", science: "Science", business: "Business",
  finance: "Finance", lifestyle: "Lifestyle", health: "Health",
  travel: "Travel", music: "Music", movies: "Movies",
  education: "Education", food: "Food", games: "Games", tv: "TV",
};

const ACTIONS: Array<{ key: string; label: string; secondary?: boolean }> = [
  { key: "read_full",   label: "Read",           secondary: true },
  { key: "save",        label: "Save",           secondary: true },
  { key: "more_like_this", label: "More like this" },
  { key: "skip",        label: "Skip" },
  { key: "not_interested", label: "Not interested" },
];

export default function NewsCard({ article, feedbackState, onFeedback }: Props) {
  const categoryLabel = CATEGORY_LABELS[article.category?.toLowerCase()] ?? article.category ?? "News";
  const score = typeof article.score === "number" ? Math.min(1, Math.max(0, article.score / 3)) : 0;
  const done = !!feedbackState;
  const reasons = Array.isArray(article.reasons) ? article.reasons.slice(0, 4) : [];
  const entities = Array.isArray(article.matched_entities) ? article.matched_entities.slice(0, 3) : [];

  const actions = [...ACTIONS];
  if (article.source) {
    actions.push({ key: "less_from_source", label: `Less from ${article.source}` });
  }

  return (
    <SurfaceCard className="flex h-full flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Pill active>{categoryLabel}</Pill>
          {article.source && (
            <span className="rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] px-3 py-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)] backdrop-blur-xl">
              {article.source}
            </span>
          )}
        </div>
        <span className="rounded-full border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.05))] px-3 py-1 text-xs text-[var(--accent-soft)] backdrop-blur-xl">
          Score {(article.score ?? 0).toFixed(3)}
        </span>
      </div>

      <div className="space-y-3">
        <h3 className="font-display text-3xl leading-tight text-[var(--foreground)] line-clamp-2">
          {article.title}
        </h3>
        <p className="text-sm leading-7 text-[var(--muted)] line-clamp-3">{article.abstract}</p>
      </div>

      {(reasons.length > 0 || entities.length > 0) && (
        <div className="space-y-2">
          {reasons.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {reasons.map((reason) => (
                <span
                  key={reason}
                  className="rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.04))] px-3 py-1.5 text-xs text-[var(--muted-strong)] backdrop-blur-xl"
                >
                  {reason}
                </span>
              ))}
            </div>
          )}
          {entities.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {entities.map((entity) => (
                <span
                  key={entity}
                  className="rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] px-3 py-1.5 text-xs text-[var(--muted)] backdrop-blur-xl"
                >
                  {entity}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="space-y-2">
        <div className="h-2 overflow-hidden rounded-full bg-white/[0.08]">
          <div
            className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),#f6e1b6)] transition-all duration-700"
            style={{ width: `${score * 100}%` }}
          />
        </div>
        <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">Confidence</p>
      </div>

      <div className="mt-auto flex flex-wrap gap-2">
        {actions.map((action) => {
          const active = feedbackState === action.key;
          return (
            <button
              key={action.key}
              onClick={() => !done && onFeedback(action.key)}
              disabled={done}
              className={[
                "inline-flex min-h-9 items-center justify-center rounded-full px-4 text-[11px] font-semibold tracking-[0.18em] uppercase transition duration-300",
                active
                  ? "border border-[rgba(255,245,220,0.28)] bg-[linear-gradient(180deg,rgba(241,215,172,0.96),rgba(230,184,108,0.82))] text-black"
                  : action.secondary
                    ? "border border-white/16 bg-[linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0.06))] text-[var(--foreground)] backdrop-blur-2xl"
                    : "border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.03))] text-[var(--muted-strong)] backdrop-blur-2xl",
                done && !active ? "opacity-40" : "",
              ].filter(Boolean).join(" ")}
            >
              {action.label}
            </button>
          );
        })}
      </div>
    </SurfaceCard>
  );
}
