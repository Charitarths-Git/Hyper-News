import Link from "next/link";
import { auth } from "@/auth";
import { BoomerangVideo } from "@/components/BoomerangVideo";
import {
  ButtonLink,
  SectionHeading,
  SkeletonCard,
  StateBlock,
  StatTile,
  SurfaceCard,
} from "@/components/ui/primitives";

const heroStats = [
  {
    label: "Signals in play",
    value: "4",
    trend: "RL, search, KG, and live mood context",
  },
  {
    label: "Guest ready",
    value: "Instant",
    trend: "The live feed works before sign-in",
  },
  {
    label: "Reasoning layer",
    value: "Visible",
    trend: "Every slate explains why it appeared",
  },
] as const;

const previewPanels = [
  {
    title: "Recommendation reasoning",
    body: "Each batch surfaces a readable explanation so the feed feels guided instead of opaque.",
  },
  {
    title: "Mood-aware pacing",
    body: "On-device affect sensing can nudge the system toward calmer, brighter, or more focused story mixes.",
  },
  {
    title: "Search with memory",
    body: "Keyword intent, profile state, and reading history are blended into the same retrieval loop.",
  },
] as const;

const productHighlights = [
  {
    title: "Live feed shell",
    body: "Refresh, feedback, search suggestions, and infinite scroll already talk to the working backend through the existing proxy routes.",
  },
  {
    title: "Readable profile state",
    body: "Dashboard and settings surfaces keep onboarding answers, interest weights, graph signals, and recent feedback legible.",
  },
  {
    title: "Calm operational states",
    body: "Loading, empty, consent, and recovery moments use the same visual language instead of falling back to bare utility UI.",
  },
] as const;

export default async function HomePage() {
  const session = await auth();
  const signedIn = Boolean(session?.user?.id);

  return (
    <div className="relative isolate min-h-screen bg-background text-foreground">
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <BoomerangVideo mode="blurred" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(6,8,12,0.18),rgba(6,8,12,0.42)_38%,rgba(6,8,12,0.68)_68%,rgba(6,8,12,0.86))]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(240,208,150,0.16),transparent_30%),radial-gradient(circle_at_bottom_left,rgba(113,154,212,0.16),transparent_28%),radial-gradient(circle_at_right,rgba(255,255,255,0.08),transparent_22%)]" />
      </div>

      <section className="relative isolate min-h-screen overflow-hidden">
        <BoomerangVideo priority mode="sharp" />
        <div className="absolute inset-0 z-10 bg-[linear-gradient(180deg,rgba(4,4,6,0.08),rgba(4,4,6,0.36)_52%,rgba(4,4,6,0.7))]" />
        <div className="absolute inset-0 z-10 bg-[radial-gradient(circle_at_top_left,rgba(230,184,108,0.16),transparent_28%),radial-gradient(circle_at_right,rgba(10,36,59,0.14),transparent_34%)]" />

        <div className="relative z-20 mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-4 sm:px-6">
          <header className="gold-ring panel-glow mac-glass-heavy rounded-[30px] px-5 py-4 sm:px-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <Link href="/" className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-[var(--accent)] text-sm font-bold text-black">
                  HN
                </div>
                <div>
                  <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-[var(--accent-soft)]">
                    HyperNews
                  </p>
                  <p className="font-display text-2xl leading-none text-[var(--foreground)]">
                    Personal news, staged cinematically.
                  </p>
                </div>
              </Link>

              <nav className="flex flex-wrap items-center gap-2">
                {[
                  { href: "/feed", label: "Feed" },
                  { href: "/search", label: "Search" },
                  { href: "/dashboard", label: "Dashboard" },
                  { href: "/profile/settings", label: "Profile" },
                ].map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.04))] px-4 py-2 text-sm text-[var(--muted-strong)] backdrop-blur-xl transition hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.14),rgba(255,255,255,0.05))]"
                  >
                    {item.label}
                  </Link>
                ))}
              </nav>

              <div className="flex flex-wrap items-center gap-3">
                {signedIn ? (
                  <>
                    <span className="rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.05))] px-4 py-2 text-sm text-[var(--muted-strong)] backdrop-blur-xl">
                      Signed in as {session?.user?.name || session?.user?.email || "reader"}
                    </span>
                    <ButtonLink href="/feed" variant="ghost">
                      Open feed
                    </ButtonLink>
                    <ButtonLink href="/dashboard">Dashboard</ButtonLink>
                  </>
                ) : (
                  <>
                    <ButtonLink href="/login" variant="ghost">
                      Sign in
                    </ButtonLink>
                    <ButtonLink href="/register">Register</ButtonLink>
                  </>
                )}
              </div>
            </div>
          </header>

          <div className="grid flex-1 items-end gap-8 py-10 lg:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)] lg:py-14">
            <div className="space-y-8">
              <div className="space-y-4">
                <p className="font-mono text-xs uppercase tracking-[0.36em] text-[var(--accent-soft)]">
                  Hyperpersonalised news intelligence
                </p>
                <h1 className="max-w-4xl font-display text-6xl leading-[0.9] text-balance text-white sm:text-7xl lg:text-[7rem]">
                  News around the world tailored for you.
                </h1>
                <p className="max-w-2xl text-base leading-8 text-[var(--muted-strong)] sm:text-lg">
                  HyperNews pairs recommendation learning, retrieval, knowledge graph context,
                  and on-device mood sensing so each session feels deliberate instead of noisy.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <ButtonLink href="/feed">Enter the live feed</ButtonLink>
                <ButtonLink href={signedIn ? "/dashboard" : "/register"} variant="secondary">
                  {signedIn ? "Open dashboard" : "Create account"}
                </ButtonLink>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                {heroStats.map((stat) => (
                  <StatTile key={stat.label} {...stat} />
                ))}
              </div>
            </div>

            <div className="grid gap-4">
              <SurfaceCard className="space-y-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm text-[var(--muted)]">Live preview</p>
                    <h2 className="font-display text-3xl text-white">Tonight&apos;s editorial posture</h2>
                  </div>
                  <span className="rounded-full bg-[var(--accent)] px-4 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-black items-center justify-center">
                    Live
                  </span>
                </div>

                <div className="grid gap-3">
                  {previewPanels.map((panel) => (
                    <div
                      key={panel.title}
                      className="rounded-[22px] border border-white/14 bg-white/[0.12] px-4 py-4 backdrop-blur-2xl"
                    >
                      <p className="text-sm font-semibold text-[var(--foreground)]">{panel.title}</p>
                      <p className="mt-2 text-sm leading-7 text-[var(--muted)]">{panel.body}</p>
                    </div>
                  ))}
                </div>
              </SurfaceCard>

              <StateBlock
                title="Route model is now split cleanly"
                description="The cinematic landing page lives at / while the working backend-connected product lives at /feed."
                tone="success"
              />
            </div>
          </div>
        </div>
      </section>

      <section className="relative isolate overflow-hidden">
        <div className="absolute inset-0 z-0 bg-[linear-gradient(180deg,rgba(4,6,9,0.42),rgba(4,6,9,0.58)_24%,rgba(4,6,9,0.74)_58%,rgba(4,6,9,0.88))]" />
        <div className="absolute inset-0 z-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.04),transparent_20%),radial-gradient(circle_at_bottom,rgba(230,184,108,0.06),transparent_28%)]" />

        <div className="relative z-10">
          <main className="mx-auto flex max-w-7xl flex-col gap-20 px-4 py-20 sm:px-6">
            <section className="space-y-8 rounded-[36px] border border-white/10 bg-[rgba(4,6,9,0.36)] px-4 py-6 backdrop-blur-[2px] sm:px-6">
              <SectionHeading
                eyebrow="Feed Preview"
                title="The feed keeps context visible, not hidden."
                description="Recommendation reasoning, feedback controls, search suggestions, and guest-to-auth transitions all already connect to the live backend."
              />

              <div className="grid gap-5 xl:grid-cols-3">
                {productHighlights.map((highlight) => (
                  <SurfaceCard key={highlight.title} className="space-y-4">
                    <p className="font-display text-3xl text-[var(--foreground)]">{highlight.title}</p>
                    <p className="text-sm leading-7 text-[var(--muted)]">{highlight.body}</p>
                  </SurfaceCard>
                ))}
              </div>
            </section>

            <section className="grid gap-8 rounded-[36px] border border-white/10 bg-[rgba(4,6,9,0.4)] px-4 py-6 backdrop-blur-[2px] sm:px-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
              <div className="space-y-6">
                <SectionHeading
                  eyebrow="Operational Surfaces"
                  title="Dashboards and settings keep the model legible."
                  description="The rest of the product stays in the same dark premium language: stacked glass panels, visible state messaging, and calm utility moments."
                />

                <div className="grid gap-4 md:grid-cols-3">
                  <SurfaceCard className="space-y-3">
                    <p className="text-sm text-[var(--muted)]">Dashboard</p>
                    <p className="font-display text-4xl text-white">Interests</p>
                    <p className="text-sm leading-7 text-[var(--muted)]">
                      Inspect interest weights, history, recent searches, graph entities, and recommendation state.
                    </p>
                  </SurfaceCard>
                  <SurfaceCard className="space-y-3">
                    <p className="text-sm text-[var(--muted)]">Profile</p>
                    <p className="font-display text-4xl text-white">Tunable</p>
                    <p className="text-sm leading-7 text-[var(--muted)]">
                      Edit onboarding answers, category priorities, region context, and affect consent in one place.
                    </p>
                  </SurfaceCard>
                  <SurfaceCard className="space-y-3">
                    <p className="text-sm text-[var(--muted)]">Mood sensing</p>
                    <p className="font-display text-4xl text-white">On-device</p>
                    <p className="text-sm leading-7 text-[var(--muted)]">
                      Face detection and expression reading stay in the browser and refresh the feed only when mood changes.
                    </p>
                  </SurfaceCard>
                </div>
              </div>

              <div className="grid gap-5">
                <SkeletonCard />
                <StateBlock
                  title="Search and empty states are designed too"
                  description="Loading, empty, consent, and exhausted moments stay polished instead of dropping into bare utility UI."
                  tone="neutral"
                />
                <StateBlock
                  title="Backend already wired"
                  description="Register, onboarding, recommend, feedback, graph, profile, reset, and suggest flows are preserved while the UI is upgraded."
                  tone="warning"
                />
              </div>
            </section>
          </main>
        </div>
      </section>
    </div>
  );
}
