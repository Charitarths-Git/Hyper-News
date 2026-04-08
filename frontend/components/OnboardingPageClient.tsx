"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import CategoryMultiSelect from "@/components/CategoryMultiSelect";
import { PageShell } from "@/components/PageShell";
import { Button, SurfaceCard } from "@/components/ui/primitives";
import {
  AGE_BUCKET_OPTIONS,
  GENDER_OPTIONS,
  OCCUPATION_OPTIONS,
  REGION_OPTIONS,
} from "@/lib/onboarding-config";
import { readJsonResponse } from "@/lib/http";
import type { Profile } from "@/lib/user-types";

interface Props {
  initialProfile: Profile;
}

const STEPS = [
  {
    key: "about",
    title: "About you",
    description: "Answer a couple of basics so the first feed is less generic.",
  },
  {
    key: "context",
    title: "Context",
    description: "Add optional work and region context for stronger cold-start matching.",
  },
  {
    key: "interests",
    title: "Interests",
    description: "Tell HyperNews what the first slate should lean toward.",
  },
] as const;

const fieldClassName =
  "themed-field mac-glass w-full rounded-[20px] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-white/30 focus:border-[var(--border-strong)]";

export default function OnboardingPageClient({ initialProfile }: Props) {
  const router = useRouter();
  const [stepIndex, setStepIndex] = useState(0);
  const [ageBucket, setAgeBucket] = useState(initialProfile.age_bucket || "");
  const [gender, setGender] = useState(initialProfile.gender || "");
  const [occupation, setOccupation] = useState(initialProfile.occupation || "");
  const [locationRegion, setLocationRegion] = useState(initialProfile.location_region || "");
  const [locationCountry, setLocationCountry] = useState(initialProfile.location_country || "");
  const [interestText, setInterestText] = useState(initialProfile.interest_text || "");
  const [topCategories, setTopCategories] = useState(initialProfile.top_categories || []);
  const [affectConsent, setAffectConsent] = useState(Boolean(initialProfile.affect_consent));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  const progress = useMemo(() => `${stepIndex + 1} / ${STEPS.length}`, [stepIndex]);
  const activeStep = STEPS[stepIndex];

  async function handleComplete() {
    if (topCategories.length === 0 && !interestText.trim()) {
      setError("Please select at least one top category or add a note about your interests before finishing.");
      return;
    }

    setSaving(true);
    setError("");

    try {
      const response = await fetch("/api/me/onboarding", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age_bucket: ageBucket,
          gender,
          occupation,
          location_region: locationRegion,
          location_country: locationCountry,
          interest_text: interestText,
          top_categories: topCategories,
          affect_consent: affectConsent,
        }),
      });
      const payload = await readJsonResponse<{ error?: string }>(response);
      if (!response.ok || payload?.error) {
        throw new Error(payload?.error || "Could not save onboarding answers.");
      }

      router.replace("/feed");
      router.refresh();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <PageShell
      eyebrow="Onboarding"
      title="Set up your first personalized feed."
      description="These answers shape cold-start recommendations before your reading behavior fully warms up the recommender."
      className="max-w-4xl"
      actions={
        <span className="rounded-full border border-white/10 bg-white/[0.08] px-4 py-2 font-mono text-xs uppercase tracking-[0.2em] text-[var(--accent-soft)]">
          Step {progress}
        </span>
      }
    >
      <SurfaceCard className="space-y-6 rounded-[32px] p-6 sm:p-8">
        <div className="space-y-4">
          <div className="flex gap-2">
            {STEPS.map((step, index) => (
              <div
                key={step.key}
                className={[
                  "h-2 flex-1 rounded-full transition",
                  index <= stepIndex
                    ? "bg-[linear-gradient(90deg,var(--accent),#f6e1b6)]"
                    : "bg-white/[0.08]",
                ].join(" ")}
              />
            ))}
          </div>

          <div>
            <p className="font-mono text-xs uppercase tracking-[0.24em] text-[var(--accent-soft)]">
              {activeStep.title}
            </p>
            <p className="mt-2 text-sm leading-7 text-[var(--muted)]">{activeStep.description}</p>
          </div>
        </div>

        {stepIndex === 0 && (
          <div className="grid gap-4 md:grid-cols-2">
            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Age bucket</span>
              <select
                value={ageBucket}
                onChange={(event) => setAgeBucket(event.target.value)}
                className={`${fieldClassName} themed-select`}
              >
                {AGE_BUCKET_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Gender</span>
              <select
                value={gender}
                onChange={(event) => setGender(event.target.value)}
                className={`${fieldClassName} themed-select`}
              >
                {GENDER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        )}

        {stepIndex === 1 && (
          <div className="grid gap-4 md:grid-cols-3">
            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Occupation</span>
              <select
                value={occupation}
                onChange={(event) => setOccupation(event.target.value)}
                className={`${fieldClassName} themed-select`}
              >
                {OCCUPATION_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Region</span>
              <select
                value={locationRegion}
                onChange={(event) => setLocationRegion(event.target.value)}
                className={`${fieldClassName} themed-select`}
              >
                {REGION_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Country</span>
              <input
                value={locationCountry}
                onChange={(event) => setLocationCountry(event.target.value)}
                placeholder="India, Japan, United States..."
                className={fieldClassName}
              />
            </label>
          </div>
        )}

        {stepIndex === 2 && (
          <div className="grid gap-5">
            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Interest notes</span>
              <textarea
                value={interestText}
                onChange={(event) => setInterestText(event.target.value)}
                placeholder="What kinds of stories do you usually follow?"
                rows={5}
                className={`${fieldClassName} min-h-[140px] resize-y`}
              />
            </label>

            <div className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Top categories</span>
              <CategoryMultiSelect selected={topCategories} onChange={setTopCategories} />
            </div>

            <label className="mac-glass flex items-start gap-3 rounded-[24px] p-4 text-sm text-[var(--muted)]">
              <input
                type="checkbox"
                checked={affectConsent}
                onChange={(event) => setAffectConsent(event.target.checked)}
                className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent accent-[var(--accent)]"
              />
              <span>
                Save consent preference for future on-device mood sensing. Camera frames stay on
                this device.
              </span>
            </label>
          </div>
        )}

        {error && (
          <div className="rounded-[20px] border border-[rgba(255,141,141,0.26)] bg-[linear-gradient(180deg,rgba(255,141,141,0.16),rgba(255,141,141,0.05))] px-4 py-3 text-sm text-[var(--danger)]">
            {error}
          </div>
        )}

        <div className="flex flex-wrap justify-between gap-3">
          <Button
            type="button"
            variant="ghost"
            onClick={() => setStepIndex((current) => Math.max(0, current - 1))}
            disabled={stepIndex === 0 || saving}
          >
            Back
          </Button>

          {stepIndex < STEPS.length - 1 ? (
            <Button
              type="button"
              onClick={() => setStepIndex((current) => Math.min(STEPS.length - 1, current + 1))}
              disabled={saving}
            >
              Continue
            </Button>
          ) : (
            <Button type="button" onClick={() => void handleComplete()} disabled={saving}>
              {saving ? "Saving..." : "Finish and open app"}
            </Button>
          )}
        </div>
      </SurfaceCard>
    </PageShell>
  );
}
