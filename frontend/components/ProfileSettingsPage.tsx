"use client";

import Link from "next/link";
import { useState } from "react";
import CategoryMultiSelect from "@/components/CategoryMultiSelect";
import { PageShell } from "@/components/PageShell";
import { Button, ButtonLink, StateBlock, StatTile, SurfaceCard } from "@/components/ui/primitives";
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

interface ProfileFormState {
  display_name: string;
  age_bucket: string;
  gender: string;
  occupation: string;
  location_region: string;
  location_country: string;
  interest_text: string;
  top_categories: string[];
  affect_consent: boolean;
}

const fieldClassName =
  "themed-field mac-glass w-full rounded-[20px] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-white/30 focus:border-[var(--border-strong)]";

export default function ProfileSettingsPage({ initialProfile }: Props) {
  const [profile, setProfile] = useState(initialProfile);
  const [form, setForm] = useState<ProfileFormState>({
    display_name: initialProfile.display_name || "",
    age_bucket: initialProfile.age_bucket || "",
    gender: initialProfile.gender || "",
    occupation: initialProfile.occupation || "",
    location_region: initialProfile.location_region || "",
    location_country: initialProfile.location_country || "",
    interest_text: initialProfile.interest_text || "",
    top_categories: initialProfile.top_categories || [],
    affect_consent: Boolean(initialProfile.affect_consent),
  });
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [saveError, setSaveError] = useState("");

  async function handleSaveProfile() {
    setSaving(true);
    setSaveError("");
    setSaveMessage("");

    try {
      const response = await fetch("/api/me/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: form.display_name,
          age_bucket: form.age_bucket,
          gender: form.gender,
          occupation: form.occupation,
          location_region: form.location_region,
          location_country: form.location_country,
          interest_text: form.interest_text,
          top_categories: form.top_categories,
          affect_consent: form.affect_consent,
        }),
      });
      const payload = await readJsonResponse<{ error?: string; profile?: Profile }>(response);
      if (!response.ok || payload?.error) {
        throw new Error(payload?.error || "Could not save your profile.");
      }

      const nextProfile = payload?.profile;
      if (nextProfile) {
        setProfile(nextProfile);
        setForm({
          display_name: nextProfile.display_name || "",
          age_bucket: nextProfile.age_bucket || "",
          gender: nextProfile.gender || "",
          occupation: nextProfile.occupation || "",
          location_region: nextProfile.location_region || "",
          location_country: nextProfile.location_country || "",
          interest_text: nextProfile.interest_text || "",
          top_categories: nextProfile.top_categories || [],
          affect_consent: Boolean(nextProfile.affect_consent),
        });
      }

      setSaveMessage("Saved. Your cold-start profile is updated.");
    } catch (nextError) {
      setSaveError(nextError instanceof Error ? nextError.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  const stats = [
    {
      label: "Positive Interactions",
      value: String(profile.total_positive_interactions ?? 0),
    },
    {
      label: "Bio Encoder",
      value: profile.has_bio_embedding ? "Ready" : "Text only",
    },
    {
      label: "Onboarding",
      value: profile.onboarding_completed ? "Complete" : "Pending",
    },
  ];

  return (
    <PageShell
      eyebrow="Profile / Settings"
      title="Edit the profile that steers the feed."
      description="Keep onboarding answers, interest notes, categories, and affect consent aligned with how you want HyperNews to adapt."
      actions={
        <div className="flex flex-wrap gap-3">
          <ButtonLink href="/feed" variant="ghost">
            Back to feed
          </ButtonLink>
          <ButtonLink href="/dashboard" variant="secondary">
            Dashboard
          </ButtonLink>
        </div>
      }
    >
      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--muted)]">
        <span>{profile.email}</span>
        <span className="rounded-full border border-white/10 bg-white/[0.08] px-3 py-1 font-mono text-xs uppercase tracking-[0.2em] text-[var(--accent-soft)]">
          Version {profile.bio_embedding_version || "pending"}
        </span>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {stats.map((stat) => (
          <StatTile key={stat.label} label={stat.label} value={stat.value} />
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_320px]">
        <SurfaceCard className="space-y-6 rounded-[32px] p-6 sm:p-8">
          <div className="space-y-2">
            <h2 className="font-display text-4xl text-[var(--foreground)]">Cold-start profile</h2>
            <p className="text-sm leading-7 text-[var(--muted)]">
              These fields matter most before behavior history fully warms up the recommender.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Display name</span>
              <input
                value={form.display_name}
                onChange={(event) =>
                  setForm((current) => ({ ...current, display_name: event.target.value }))
                }
                placeholder="What should we call you?"
                className={`${fieldClassName} themed-select`}
              />
            </label>

            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Age bucket</span>
              <select
                value={form.age_bucket}
                onChange={(event) =>
                  setForm((current) => ({ ...current, age_bucket: event.target.value }))
                }
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
                value={form.gender}
                onChange={(event) =>
                  setForm((current) => ({ ...current, gender: event.target.value }))
                }
                className={`${fieldClassName} themed-select`}
              >
                {GENDER_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
              <span>Occupation</span>
              <select
                value={form.occupation}
                onChange={(event) =>
                  setForm((current) => ({ ...current, occupation: event.target.value }))
                }
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
                value={form.location_region}
                onChange={(event) =>
                  setForm((current) => ({ ...current, location_region: event.target.value }))
                }
                className={fieldClassName}
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
                value={form.location_country}
                onChange={(event) =>
                  setForm((current) => ({ ...current, location_country: event.target.value }))
                }
                placeholder="India, Japan, United States..."
                className={fieldClassName}
              />
            </label>
          </div>

          <div className="grid gap-2 text-sm text-[var(--muted-strong)]">
            <span>Top categories</span>
            <CategoryMultiSelect
              selected={form.top_categories}
              onChange={(nextTopCategories) =>
                setForm((current) => ({ ...current, top_categories: nextTopCategories }))
              }
            />
          </div>

          <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
            <span>Interest notes</span>
            <textarea
              value={form.interest_text}
              onChange={(event) =>
                setForm((current) => ({ ...current, interest_text: event.target.value }))
              }
              rows={5}
              placeholder="Tell HyperNews what kinds of stories you like following."
              className={`${fieldClassName} min-h-[140px] resize-y`}
            />
          </label>

          <label className="mac-glass flex items-start gap-3 rounded-[24px] p-4 text-sm text-[var(--muted)]">
            <input
              type="checkbox"
              checked={form.affect_consent}
              onChange={(event) =>
                setForm((current) => ({ ...current, affect_consent: event.target.checked }))
              }
              className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent accent-[var(--accent)]"
            />
            <span>
              Save consent for future on-device affect sensing. This is stored as a preference
              only.
            </span>
          </label>

          <div className="flex flex-wrap justify-between gap-3">
            <Link href="/dashboard" className="text-sm text-[var(--accent-soft)]">
              View current dashboard state
            </Link>
            <Button type="button" onClick={handleSaveProfile} disabled={saving}>
              {saving ? "Saving..." : "Save profile"}
            </Button>
          </div>
        </SurfaceCard>

        <div className="grid gap-4">
          <StateBlock
            title="Save confirmation"
            description={
              saveMessage ||
              "Saved profile changes appear here so the next refresh feels intentional."
            }
            tone={saveMessage ? "success" : "neutral"}
          />
          <StateBlock
            title="Save errors"
            description={
              saveError ||
              "Temporary failures and validation issues surface here without displacing the form."
            }
            tone={saveError ? "danger" : "warning"}
          />
        </div>
      </div>
    </PageShell>
  );
}
