"use client";

import type { FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { signIn } from "next-auth/react";
import { BoomerangVideo } from "@/components/BoomerangVideo";
import { Button, ButtonLink, SurfaceCard } from "@/components/ui/primitives";
import { readJsonResponse } from "@/lib/http";

const fieldClassName =
  "mac-glass w-full rounded-[20px] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-white/30 focus:border-[var(--border-strong)]";

export default function RegisterPageClient() {
  const router = useRouter();
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      const registerResponse = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          display_name: displayName,
          email,
          password,
        }),
      });
      const registerPayload = await readJsonResponse<{ error?: string }>(registerResponse);
      if (!registerResponse.ok || registerPayload?.error) {
        throw new Error(registerPayload?.error || "Could not create your account.");
      }

      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      });
      if (result?.error) {
        throw new Error("Your account was created, but sign-in failed.");
      }

      router.replace("/onboarding");
      router.refresh();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "Something went wrong.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="relative isolate min-h-screen overflow-hidden px-4 py-6 sm:px-6">
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <BoomerangVideo mode="blurred" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(4,6,9,0.52),rgba(4,6,9,0.76))]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(240,208,150,0.12),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(113,154,212,0.12),transparent_26%)]" />
      </div>

      <div className="relative z-10 mx-auto flex min-h-screen max-w-7xl flex-col">
        <div className="flex justify-end py-2">
          <ButtonLink href="/" variant="ghost">
            Back to landing
          </ButtonLink>
        </div>

        <div className="flex flex-1 items-center justify-center py-10">
          <SurfaceCard className="w-full max-w-xl space-y-6 rounded-[32px] p-7 sm:p-8">
            <div className="space-y-3">
              <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--accent-soft)]">
                HyperNews
              </p>
              <h1 className="font-display text-5xl leading-none text-[var(--foreground)]">
                Create your account.
              </h1>
              <p className="text-sm leading-7 text-[var(--muted)]">
                Account creation comes first. Onboarding follows next, then the live feed opens.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="grid gap-4">
              <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
                <span>Display name</span>
                <input
                  value={displayName}
                  onChange={(event) => setDisplayName(event.target.value)}
                  placeholder="What should we call you?"
                  className={fieldClassName}
                />
              </label>

              <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
                <span>Email</span>
                <input
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="name@example.com"
                  autoComplete="email"
                  className={fieldClassName}
                  required
                />
              </label>

              <label className="grid gap-2 text-sm text-[var(--muted-strong)]">
                <span>Password</span>
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="At least 8 characters"
                  autoComplete="new-password"
                  className={fieldClassName}
                  required
                />
              </label>

              {error && (
                <div className="rounded-[20px] border border-[rgba(255,141,141,0.26)] bg-[linear-gradient(180deg,rgba(255,141,141,0.16),rgba(255,141,141,0.05))] px-4 py-3 text-sm text-[var(--danger)]">
                  {error}
                </div>
              )}

              <Button type="submit" disabled={submitting} className="mt-2">
                {submitting ? "Creating account..." : "Create account"}
              </Button>
            </form>

            <div className="text-sm text-[var(--muted)]">
              Already have an account?{" "}
              <Link href="/login" className="font-semibold text-[var(--accent-soft)]">
                Sign in here
              </Link>
            </div>
          </SurfaceCard>
        </div>
      </div>
    </div>
  );
}
