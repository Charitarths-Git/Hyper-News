"use client";

import type { FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { signIn } from "next-auth/react";
import { BoomerangVideo } from "@/components/BoomerangVideo";
import { Button, ButtonLink, SurfaceCard } from "@/components/ui/primitives";

const fieldClassName =
  "mac-glass w-full rounded-[20px] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-white/30 focus:border-[var(--border-strong)]";

export default function LoginPageClient() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");

    try {
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      });
      if (result?.error) {
        throw new Error("Invalid email or password.");
      }
      router.replace("/feed");
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
                Sign in to your feed.
              </h1>
              <p className="text-sm leading-7 text-[var(--muted)]">
                Pick up your personalized feed, dashboard, saved profile state, and search history.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="grid gap-4">
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
                  autoComplete="current-password"
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
                {submitting ? "Signing in..." : "Sign in"}
              </Button>
            </form>

            <div className="text-sm text-[var(--muted)]">
              Need an account?{" "}
              <Link href="/register" className="font-semibold text-[var(--accent-soft)]">
                Register here
              </Link>
            </div>
          </SurfaceCard>
        </div>
      </div>
    </div>
  );
}
