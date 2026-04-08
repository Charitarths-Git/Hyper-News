"use client";

import { MAX_TOP_CATEGORIES, TOP_CATEGORY_OPTIONS } from "@/lib/onboarding-config";

interface Props {
  selected: string[];
  onChange: (next: string[]) => void;
  maxSelected?: number;
}

export default function CategoryMultiSelect({
  selected,
  onChange,
  maxSelected = MAX_TOP_CATEGORIES,
}: Props) {
  const selectedSet = new Set(selected);
  const selectionLimitReached = selected.length >= maxSelected;

  function toggleCategory(value: string) {
    if (selectedSet.has(value)) {
      onChange(selected.filter((item) => item !== value));
      return;
    }
    if (selectionLimitReached) {
      return;
    }
    onChange([...selected, value]);
  }

  return (
    <div className="grid gap-4">
      <div className="flex items-center justify-between gap-3 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
        <span>Choose up to {maxSelected} categories</span>
        <span className="font-mono text-[var(--accent-soft)]">
          {selected.length}/{maxSelected}
        </span>
      </div>

      <div className="flex flex-wrap gap-2">
        {TOP_CATEGORY_OPTIONS.map((option) => {
          const active = selectedSet.has(option.value);
          const disabled = !active && selectionLimitReached;

          return (
            <button
              key={option.value}
              type="button"
              aria-pressed={active}
              onClick={() => toggleCategory(option.value)}
              disabled={disabled}
              className={[
                "inline-flex items-center rounded-full border px-3 py-2 text-xs font-medium uppercase tracking-[0.14em] transition",
                active
                  ? "border-[rgba(255,245,220,0.24)] bg-[linear-gradient(180deg,rgba(241,215,172,0.92),rgba(230,184,108,0.78))] text-black shadow-[0_10px_30px_rgba(230,184,108,0.18)]"
                  : "border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.05))] text-[var(--muted-strong)] backdrop-blur-xl",
                disabled ? "cursor-not-allowed opacity-40" : "cursor-pointer hover:bg-white/[0.16]",
              ]
                .filter(Boolean)
                .join(" ")}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      <div className="flex min-h-10 flex-wrap gap-2">
        {selected.length > 0 ? (
          selected.map((value) => {
            const option = TOP_CATEGORY_OPTIONS.find((item) => item.value === value);
            return (
              <span
                key={value}
                className="rounded-full border border-white/10 bg-white/[0.08] px-3 py-1.5 text-xs text-[var(--muted-strong)]"
              >
                {option?.label || value}
              </span>
            );
          })
        ) : (
          <span className="text-sm text-[var(--muted)]">No categories selected yet.</span>
        )}
      </div>
    </div>
  );
}
