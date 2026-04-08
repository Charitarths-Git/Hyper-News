"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Button, Pill, RangeSlider, cx } from "@/components/ui/primitives";

interface Mood {
  key: string;
  emoji: string;
  label: string;
}

interface Props {
  surface: "feed" | "search";
  isAuthenticated: boolean;
  userLabel: string;
  mood: string;
  setMood: (v: string) => void;
  moods: Mood[];
  query: string;
  setQuery: (v: string) => void;
  suggestions: string[];
  onSearch: () => void;
  onRefresh: () => void;
  onResetProfile: () => void;
  onSignOut: () => void;
  onSignIn: () => void;
  onRegister: () => void;
  loading: boolean;
  mode: string;
  exploreFocus: number;
  setExploreFocus: (v: number) => void;
  affectEnabled: boolean;
  setAffectEnabled: () => void;
  affectSensor?: React.ReactNode;
}

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/feed", label: "Feed" },
  { href: "/search", label: "Search" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/profile/settings", label: "Profile" },
];

const MODE_LABELS: Record<string, string> = {
  rag: "RAG",
  rl: "RL",
  cold_start: "Cold Start",
};

export default function ContextBar({
  surface,
  isAuthenticated,
  userLabel,
  mood,
  setMood,
  moods,
  query,
  setQuery,
  suggestions,
  onSearch,
  onRefresh,
  onResetProfile,
  onSignOut,
  onSignIn,
  onRegister,
  loading,
  mode,
  exploreFocus,
  setExploreFocus,
  affectEnabled,
  setAffectEnabled,
  affectSensor,
}: Props) {
  const pathname = usePathname();
  const isActive = (href: string) => {
    if (!pathname) {
      return false;
    }
    if (href === "/profile/settings") {
      return pathname === "/profile" || pathname.startsWith("/profile/");
    }
    return pathname === href;
  };

  return (
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
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={cx(
                "rounded-full px-4 py-2 text-sm transition",
                isActive(item.href)
                  ? "bg-[var(--accent)] text-black"
                  : "border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.1),rgba(255,255,255,0.04))] text-[var(--muted-strong)] backdrop-blur-xl hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.14),rgba(255,255,255,0.05))]",
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="flex flex-wrap items-center gap-3">
          {mode && (
            <span className="rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 font-mono text-xs uppercase tracking-[0.2em] text-[var(--accent-soft)] backdrop-blur-xl">
              {MODE_LABELS[mode] ?? mode}
            </span>
          )}
          {isAuthenticated ? (
            <>
              <span className="rounded-full border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.05))] px-4 py-2 text-sm text-[var(--muted-strong)] backdrop-blur-xl">
                {userLabel}
              </span>
              <Button
                variant="ghost"
                onClick={onResetProfile}
                disabled={loading}
                className="px-4 py-2 text-xs"
              >
                Reset
              </Button>
              <Button variant="ghost" onClick={onSignOut} className="px-4 py-2 text-xs">
                Sign out
              </Button>
            </>
          ) : (
            <>
              <Button variant="ghost" onClick={onSignIn} className="px-4 py-2 text-xs">
                Sign in
              </Button>
              <Button onClick={onRegister} className="px-4 py-2 text-xs">
                Register
              </Button>
              <Button
                variant="ghost"
                onClick={onResetProfile}
                disabled={loading}
                className="px-4 py-2 text-xs"
              >
                Reset session
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,0.8fr)]">
        <div className="panel-glow mac-glass rounded-[26px] p-4">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              onSearch();
            }}
            className="flex flex-col gap-3 md:flex-row"
          >
            <div className="flex-1">
              <input
                id="news-search"
                list="hypernews-search-suggestions"
                name="newsSearch"
                type="text"
                placeholder="Trace a topic, source, or mood..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                className="mac-glass w-full rounded-[20px] px-4 py-3 text-sm text-[var(--foreground)] outline-none transition placeholder:text-white/30 focus:border-[var(--border-strong)]"
              />
              <datalist id="hypernews-search-suggestions">
                {suggestions.map((suggestion) => (
                  <option key={suggestion} value={suggestion} />
                ))}
              </datalist>
            </div>
            <div className="flex gap-2 md:self-end">
              <Button type="submit" disabled={loading} className="px-5 py-2 text-xs">
                {loading && surface === "search" ? "Searching..." : "Search"}
              </Button>
              <Button
                variant="secondary"
                type="button"
                onClick={onRefresh}
                disabled={loading}
                className="px-5 py-2 text-xs"
              >
                {loading && surface === "feed" ? "Loading..." : "Refresh"}
              </Button>
            </div>
          </form>
          {suggestions.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {suggestions.slice(0, 6).map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => {
                    setQuery(item);
                    onSearch();
                  }}
                  className="rounded-full border border-white/14 bg-white/[0.14] px-3 py-2 text-xs text-[var(--muted)] backdrop-blur-2xl transition hover:bg-white/[0.2]"
                >
                  {item}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(200px,0.8fr)] lg:grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(200px,0.8fr)]">
          <div className="panel-glow mac-glass rounded-[26px] p-4">
            <p className="mb-3 text-sm text-[var(--muted-strong)]">Mood</p>
            <div className="flex flex-wrap gap-2">
              {moods.map((entry) => (
                <Pill key={entry.key} active={mood === entry.key} onClick={() => setMood(entry.key)}>
                  {entry.emoji} {entry.label}
                </Pill>
              ))}
              <button
                onClick={setAffectEnabled}
                className={cx(
                  "inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium tracking-[0.12em] uppercase backdrop-blur-xl transition",
                  affectEnabled
                    ? "border-[rgba(122,215,168,0.3)] bg-[linear-gradient(180deg,rgba(122,215,168,0.2),rgba(122,215,168,0.08))] text-[var(--success)]"
                    : "border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.12),rgba(255,255,255,0.05))] text-[var(--muted-strong)]",
                )}
              >
                {affectEnabled ? "Camera On" : "Auto Mood"}
              </button>
              {affectSensor}
            </div>
          </div>

          <RangeSlider label="Explore / Focus" value={exploreFocus} onChange={setExploreFocus} />
        </div>
      </div>
    </header>
  );
}
