"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { signOut, useSession } from "next-auth/react";

import AffectSensor from "@/components/AffectSensor";
import ContextBar from "@/components/ContextBar";
import ExplanationBanner from "@/components/ExplanationBanner";
import NewsCard from "@/components/NewsCard";
import { apiUrl } from "@/lib/api";
import { readJsonResponse } from "@/lib/http";

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

interface RecommendResponse {
  articles: Article[];
  explanation: string;
  mode: string;
  request_id?: string;
  error?: string;
}

interface SuggestResponse {
  suggestions: string[];
}

interface FetchOptions {
  append?: boolean;
  excludeIds?: string[];
  batchSize?: number;
  queryOverride?: string | null;
  retryOnFailure?: boolean;
  retryCount?: number;
}

interface Props {
  surface: "feed" | "search";
  initialQuery?: string;
}

const MOODS = [
  { key: "neutral", emoji: "Calm", label: "Neutral" },
  { key: "curious", emoji: "Explore", label: "Curious" },
  { key: "happy", emoji: "Bright", label: "Happy" },
  { key: "stressed", emoji: "Light", label: "Stressed" },
  { key: "tired", emoji: "Easy", label: "Tired" },
];

const PAGE_SIZE = 8;
const DEFAULT_MOOD = "neutral";
const DEFAULT_EXPLORE_FOCUS = 55;
const GUEST_USER_STORAGE_KEY = "hypernews_guest_user_id";
const MOOD_STORAGE_KEY = "hypernews_mood";
const EXPLORE_STORAGE_KEY = "hypernews_explore_focus";

function normalizeArticleId(value: unknown): string {
  return String(value ?? "").trim();
}

function createGuestUserId(): string {
  return `guest_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

function articleIdsFromList(items: Article[]): string[] {
  return items
    .map((article) => normalizeArticleId(article.news_id))
    .filter(Boolean);
}

function dedupeArticles(items: Article[], seenIds: Set<string> = new Set()): Article[] {
  const unique: Article[] = [];
  for (const article of items) {
    const newsId = normalizeArticleId(article.news_id);
    if (!newsId || seenIds.has(newsId)) {
      continue;
    }
    seenIds.add(newsId);
    unique.push(article);
  }
  return unique;
}

export default function RecommendationSurface({ surface, initialQuery = "" }: Props) {
  const router = useRouter();
  const normalizedInitialQuery = initialQuery.trim();
  const { data: session, status } = useSession();
  const [guestUserId, setGuestUserId] = useState("");
  const [preferencesReady, setPreferencesReady] = useState(false);
  const [mood, setMood] = useState(DEFAULT_MOOD);
  const [exploreFocus, setExploreFocus] = useState(DEFAULT_EXPLORE_FOCUS);
  const [queryInput, setQueryInput] = useState(normalizedInitialQuery);
  const [activeQuery, setActiveQuery] = useState(surface === "search" ? normalizedInitialQuery : "");
  const [articles, setArticles] = useState<Article[]>([]);
  const [explanation, setExplanation] = useState("");
  const [mode, setMode] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [hasLoadedSurface, setHasLoadedSurface] = useState(false);
  const [feedbackMap, setFeedbackMap] = useState<Record<string, string>>({});
  const [toastMsg, setToastMsg] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [lastRequestId, setLastRequestId] = useState("");
  const [affectEnabled, setAffectEnabled] = useState(false);
  const [showConsentModal, setShowConsentModal] = useState(false);
  const [affectConsented, setAffectConsented] = useState(false);
  const retryTimerRef = useRef<number | null>(null);
  const requestIdRef = useRef(0);
  const loadMoreSentinelRef = useRef<HTMLDivElement | null>(null);
  const recommendAbortRef = useRef<AbortController | null>(null);
  const articlesRef = useRef<Article[]>([]);
  const moodRef = useRef(mood);
  const exploreFocusRef = useRef(exploreFocus);
  const activeQueryRef = useRef(activeQuery);

  const isAuthenticated = status === "authenticated" && !!session?.user?.id;
  const userId = isAuthenticated ? String(session?.user?.id || "") : guestUserId;
  const userLabel = isAuthenticated
    ? session?.user?.name || session?.user?.email || userId
    : "Guest session";

  useEffect(() => {
    moodRef.current = mood;
  }, [mood]);

  useEffect(() => {
    exploreFocusRef.current = exploreFocus;
  }, [exploreFocus]);

  useEffect(() => {
    activeQueryRef.current = activeQuery;
  }, [activeQuery]);

  useEffect(() => {
    articlesRef.current = articles;
  }, [articles]);

  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current !== null) {
      window.clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  const showToast = useCallback((msg: string) => {
    setToastMsg(msg);
    window.setTimeout(() => setToastMsg(""), 2500);
  }, []);

  useEffect(() => {
    return () => {
      clearRetryTimer();
      recommendAbortRef.current?.abort();
    };
  }, [clearRetryTimer]);

  useEffect(() => {
    if (typeof window === "undefined" || isAuthenticated) {
      return;
    }
    const existingGuestUserId = window.localStorage.getItem(GUEST_USER_STORAGE_KEY);
    if (existingGuestUserId) {
      setGuestUserId(existingGuestUserId);
      return;
    }
    const nextGuestUserId = createGuestUserId();
    window.localStorage.setItem(GUEST_USER_STORAGE_KEY, nextGuestUserId);
    setGuestUserId(nextGuestUserId);
  }, [isAuthenticated]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const storedMood = window.localStorage.getItem(MOOD_STORAGE_KEY);
    if (storedMood && MOODS.some((entry) => entry.key === storedMood)) {
      setMood(storedMood);
    }

    const storedExploreFocus = Number(window.localStorage.getItem(EXPLORE_STORAGE_KEY));
    if (Number.isFinite(storedExploreFocus)) {
      setExploreFocus(Math.max(0, Math.min(100, storedExploreFocus)));
    }

    setPreferencesReady(true);
  }, []);

  useEffect(() => {
    if (!preferencesReady || typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(MOOD_STORAGE_KEY, mood);
    window.localStorage.setItem(EXPLORE_STORAGE_KEY, String(exploreFocus));
  }, [exploreFocus, mood, preferencesReady]);

  useEffect(() => {
    const nextQuery = normalizedInitialQuery;
    setQueryInput(nextQuery);
    setActiveQuery(surface === "search" ? nextQuery : "");
    requestIdRef.current += 1;
    setArticles([]);
    setExplanation("");
    setMode("");
    setFeedbackMap({});
    setHasMore(true);
    setHasLoadedSurface(false);
    setLastRequestId("");
    clearRetryTimer();
  }, [clearRetryTimer, normalizedInitialQuery, surface]);

  useEffect(() => {
    if (!userId || queryInput.trim().length < 2) {
      setSuggestions([]);
      return;
    }

    const controller = new AbortController();
    const timeout = window.setTimeout(async () => {
      try {
        const suggestUrl = isAuthenticated
          ? `${apiUrl("/search/suggest")}?q=${encodeURIComponent(queryInput)}&limit=8`
          : `${apiUrl("/public/search/suggest")}?q=${encodeURIComponent(queryInput)}&guest_id=${encodeURIComponent(userId)}&limit=8`;
        const response = await fetch(suggestUrl, { signal: controller.signal });
        if (!response.ok) {
          return;
        }
        const data: SuggestResponse = await response.json();
        setSuggestions(Array.isArray(data.suggestions) ? data.suggestions : []);
      } catch {
        /* no-op */
      }
    }, 120);

    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [isAuthenticated, queryInput, userId]);

  const fetchRecommendations = useCallback(
    async ({
      append = false,
      excludeIds = [],
      batchSize = PAGE_SIZE,
      queryOverride,
      retryOnFailure = false,
      retryCount = 0,
    }: FetchOptions = {}) => {
      if (!userId) {
        return false;
      }

      const queryValue = surface === "search"
        ? String(queryOverride ?? activeQueryRef.current ?? "").trim()
        : "";

      if (surface === "search" && !queryValue) {
        setArticles([]);
        setExplanation("");
        setMode("");
        setHasLoadedSurface(true);
        setHasMore(false);
        return false;
      }

      const normalizedExcludeIds = excludeIds
        .map((value) => normalizeArticleId(value))
        .filter(Boolean);
      const requestId = append ? requestIdRef.current : ++requestIdRef.current;
      const requestToken = `req_${Date.now()}_${requestId}`;
      const recommendUrl = isAuthenticated ? apiUrl("/recommend") : apiUrl("/public/recommend");

      if (append) {
        setLoadingMore(true);
      } else {
        clearRetryTimer();
        recommendAbortRef.current?.abort();
        setLoading(true);
      }

      try {
        const controller = new AbortController();
        recommendAbortRef.current = controller;
        const res = await fetch(recommendUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            ...(isAuthenticated ? {} : { guest_id: userId }),
            request_id: requestToken,
            mood: moodRef.current,
            query: surface === "search" ? queryValue : null,
            n: batchSize,
            exclude_ids: normalizedExcludeIds,
            surface,
            explore_focus: exploreFocusRef.current,
          }),
        });

        const data = await readJsonResponse<RecommendResponse>(res);
        if (!res.ok || data?.error) {
          throw new Error(data?.error || `Recommend request failed with ${res.status}`);
        }
        if (requestId !== requestIdRef.current) {
          return false;
        }

        const nextArticles = Array.isArray(data.articles) ? data.articles : [];
        const uniqueArticles = append
          ? dedupeArticles(nextArticles, new Set(articleIdsFromList(articlesRef.current)))
          : dedupeArticles(nextArticles);

        setLastRequestId(String(data.request_id || requestToken));

        if (append) {
          setArticles((prev) => [...prev, ...uniqueArticles]);
          if (data.mode) {
            setMode(data.mode);
          }
          setHasMore(uniqueArticles.length > 0 && nextArticles.length >= batchSize);
        } else {
          if (
            normalizedExcludeIds.length > 0 &&
            uniqueArticles.length === 0 &&
            articlesRef.current.length > 0
          ) {
            setHasMore(false);
            showToast("No fresh unseen stories are available right now.");
            setHasLoadedSurface(true);
            return false;
          }

          setArticles(uniqueArticles);
          setExplanation(data.explanation || "");
          setMode(data.mode || "");
          setToastMsg("");
          setFeedbackMap((prev) => {
            const next: Record<string, string> = {};
            for (const article of uniqueArticles) {
              const existing = prev[normalizeArticleId(article.news_id)];
              if (existing) {
                next[normalizeArticleId(article.news_id)] = existing;
              }
            }
            return next;
          });
          setHasMore(uniqueArticles.length > 0 && nextArticles.length >= batchSize);
        }

        setHasLoadedSurface(true);
        return uniqueArticles.length > 0;
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          return false;
        }
        if (requestId !== requestIdRef.current) {
          return false;
        }

        if (append) {
          console.error("Load-more request failed", error);
          showToast("Could not load more articles right now.");
        } else {
          console.error("Recommend request failed", error);
          setToastMsg(
            error instanceof Error && error.message
              ? error.message
              : "Cannot reach backend.",
          );
          if (retryOnFailure && retryCount < 1) {
            retryTimerRef.current = window.setTimeout(() => {
              void fetchRecommendations({
                batchSize,
                queryOverride: surface === "search" ? queryValue : null,
                retryOnFailure: true,
                retryCount: retryCount + 1,
              });
            }, 2000);
          }
        }

        setHasLoadedSurface(true);
        return false;
      } finally {
        if (append) {
          setLoadingMore(false);
        } else if (requestId === requestIdRef.current) {
          setLoading(false);
        }
      }
    },
    [clearRetryTimer, isAuthenticated, showToast, surface, userId],
  );

  useEffect(() => {
    if (status === "loading" || !userId || !preferencesReady) {
      return;
    }

    if (surface === "search") {
      if (!activeQueryRef.current) {
        setArticles([]);
        setExplanation("");
        setMode("");
        setHasLoadedSurface(true);
        setHasMore(false);
        return;
      }
      void fetchRecommendations({
        queryOverride: activeQueryRef.current,
        retryOnFailure: true,
      });
      return;
    }

    void fetchRecommendations({ retryOnFailure: true });
  }, [fetchRecommendations, preferencesReady, status, surface, userId]);

  const submitSearch = useCallback(() => {
    const nextQuery = queryInput.trim();
    if (!nextQuery) {
      if (surface === "search") {
        router.push("/");
      } else {
        showToast("Enter a search topic to open the search page.");
      }
      return;
    }

    if (surface === "search") {
      if (nextQuery === activeQueryRef.current) {
        setHasMore(true);
        void fetchRecommendations({
          queryOverride: nextQuery,
          retryOnFailure: true,
        });
        return;
      }
    }

    router.push(`/search?q=${encodeURIComponent(nextQuery)}`);
  }, [fetchRecommendations, queryInput, router, showToast, surface]);

  const refreshCurrentSurface = useCallback(() => {
    if (surface === "search" && !activeQueryRef.current) {
      return;
    }
    setHasMore(true);
    void fetchRecommendations({
      queryOverride: surface === "search" ? activeQueryRef.current : null,
      excludeIds: articleIdsFromList(articlesRef.current),
      retryOnFailure: true,
    });
  }, [fetchRecommendations, surface]);

  const loadMoreArticles = useCallback(() => {
    if (
      !userId
      || loading
      || loadingMore
      || !hasMore
      || articlesRef.current.length === 0
      || (surface === "search" && !activeQueryRef.current)
    ) {
      return;
    }

    void fetchRecommendations({
      append: true,
      queryOverride: surface === "search" ? activeQueryRef.current : null,
      excludeIds: articleIdsFromList(articlesRef.current),
      batchSize: PAGE_SIZE,
    });
  }, [fetchRecommendations, hasMore, loading, loadingMore, surface, userId]);

  useEffect(() => {
    const sentinel = loadMoreSentinelRef.current;
    if (!sentinel || !hasMore || loading || loadingMore || articles.length === 0) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          loadMoreArticles();
        }
      },
      { rootMargin: "500px 0px" },
    );

    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [articles.length, hasMore, loadMoreArticles, loading, loadingMore]);

  const sendFeedback = async (article: Article, position: number, action: string) => {
    const normalizedArticleId = normalizeArticleId(article.news_id);
    if (!normalizedArticleId || !userId) {
      return;
    }

    const previousArticles = articlesRef.current;
    const previousFeedback = feedbackMap[normalizedArticleId];
    const isRemovalAction =
      action === "skip" || action === "not_interested" || action === "less_from_source";

    setFeedbackMap((prev) => ({ ...prev, [normalizedArticleId]: action }));
    if (isRemovalAction) {
      setArticles((prev) =>
        prev.filter((entry) => normalizeArticleId(entry.news_id) !== normalizedArticleId),
      );
    }

    const actionToastMap: Record<string, string> = {
      read_full: "Preference saved for upcoming stories",
      save: "Saved for future recommendations",
      more_like_this: "We will lean further into this topic",
      skip: "Skipped. Upcoming stories will adjust",
      not_interested: "We will downrank similar stories",
      less_from_source: `We will reduce stories from ${article.source || "this source"}`,
    };
    showToast(actionToastMap[action] || "Feedback saved");

    try {
      const response = await fetch(
        isAuthenticated ? apiUrl("/feedback") : apiUrl("/public/feedback"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...(isAuthenticated ? {} : { guest_id: userId }),
            request_id: lastRequestId,
            impression_id: normalizedArticleId,
            article_id: normalizedArticleId,
            action,
            position,
            query_text: surface === "search" ? activeQueryRef.current : "",
            source_feedback: article.source || "",
          }),
        },
      );
      if (!response.ok) {
        throw new Error(`Feedback request failed with ${response.status}`);
      }
    } catch (error) {
      console.error("Feedback request failed", error);
      if (isRemovalAction) {
        setArticles(previousArticles);
      }
      setFeedbackMap((prev) => {
        const next = { ...prev };
        if (previousFeedback) {
          next[normalizedArticleId] = previousFeedback;
        } else {
          delete next[normalizedArticleId];
        }
        return next;
      });
      showToast("Could not save feedback right now.");
    }
  };

  // Load persisted consent from localStorage (covers both guests and auth'd users)
  useEffect(() => {
    if (typeof window === "undefined") return;
    const stored = window.localStorage.getItem("hypernews_affect_consent");
    if (stored === "true") setAffectConsented(true);
  }, []);

  const handleToggleAffect = useCallback(() => {
    if (affectEnabled) {
      setAffectEnabled(false);
      return;
    }
    if (affectConsented) {
      setAffectEnabled(true);
    } else {
      setShowConsentModal(true);
    }
  }, [affectEnabled, affectConsented]);

  const handleConsentGrant = useCallback(async () => {
    setShowConsentModal(false);
    setAffectConsented(true);
    setAffectEnabled(true);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("hypernews_affect_consent", "true");
    }
    // Persist to backend profile for authenticated users
    if (isAuthenticated) {
      try {
        await fetch("/api/me/profile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ affect_consent: true }),
        });
      } catch {
        /* non-critical - consent already stored in localStorage */
      }
    }
  }, [isAuthenticated]);

  const handleMoodDetected = useCallback(
    (detectedMood: string) => {
      if (detectedMood === moodRef.current) return;
      // Update ref immediately so fetchRecommendations reads the new mood
      moodRef.current = detectedMood;
      setMood(detectedMood);
      // Full fresh fetch with no excludeIds so the explanation banner also reflects the new mood.
      void fetchRecommendations({
        queryOverride: surface === "search" ? activeQueryRef.current : null,
      });
    },
    [fetchRecommendations, surface],
  );

  const resetProfile = async () => {
    clearRetryTimer();
    requestIdRef.current += 1;

    try {
      await fetch(isAuthenticated ? apiUrl("/reset") : apiUrl("/public/reset"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(isAuthenticated ? {} : { guest_id: userId }),
      });
    } catch {
      /* silent fail */
    }

    articlesRef.current = [];
    setArticles([]);
    setExplanation("");
    setMode("");
    setFeedbackMap({});
    setHasLoadedSurface(false);
    setHasMore(true);
    if (surface === "search") {
      setQueryInput(activeQueryRef.current);
    }
    setSuggestions([]);
    setLastRequestId("");
    showToast("Profile memory reset");

    if (!userId) {
      return;
    }

    if (surface === "search" && !activeQueryRef.current) {
      setHasLoadedSurface(true);
      return;
    }

    void fetchRecommendations({
      queryOverride: surface === "search" ? activeQueryRef.current : null,
      retryOnFailure: true,
    });
  };

  const showWelcome = surface === "feed" && !loading && articles.length === 0 && !hasLoadedSurface;
  const showSearchPrompt = surface === "search" && !loading && articles.length === 0 && hasLoadedSurface && !activeQuery;
  const showEmptyState = !loading && articles.length === 0 && hasLoadedSurface && !showSearchPrompt;

  if (status === "loading" || (!isAuthenticated && !guestUserId) || !preferencesReady) {
    return (
      <div className="relative isolate min-h-screen overflow-hidden px-4 py-4 sm:px-6">
        <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
          <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(4,6,9,0.8),rgba(4,6,9,0.95))]" />
        </div>
        <div className="relative z-10 flex min-h-screen items-center justify-center">
          <div className="panel-glow mac-glass rounded-[28px] p-8 text-center space-y-3">
            <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--accent-soft)]">HyperNews</p>
            <p className="font-display text-3xl text-[var(--foreground)]">Loading...</p>
          </div>
        </div>
      </div>
    );
  }

  const header = (
    <ContextBar
      surface={surface}
      isAuthenticated={isAuthenticated}
      userLabel={userLabel}
      mood={mood}
      setMood={setMood}
      moods={MOODS}
      query={queryInput}
      setQuery={setQueryInput}
      suggestions={suggestions}
      onSearch={submitSearch}
      onRefresh={refreshCurrentSurface}
      onResetProfile={resetProfile}
      onSignOut={() => void signOut({ callbackUrl: "/login" })}
      onSignIn={() => router.push("/login")}
      onRegister={() => router.push("/register")}
      loading={loading}
      mode={mode}
      exploreFocus={exploreFocus}
      setExploreFocus={setExploreFocus}
      affectEnabled={affectEnabled}
      setAffectEnabled={handleToggleAffect}
      affectSensor={<AffectSensor enabled={affectEnabled} onMoodDetected={handleMoodDetected} />}
    />
  );

  return (
    <div className="relative isolate min-h-screen overflow-hidden px-4 py-4 sm:px-6">
      {/* Background */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <video
          autoPlay muted playsInline loop
          className="absolute inset-0 h-full w-full object-cover scale-110 blur-[20px] saturate-150 brightness-[0.82]"
        >
          <source src="https://res.cloudinary.com/dfonotyfb/video/upload/v1775585556/dds3_1_rqhg7x.mp4" type="video/mp4" />
        </video>
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(4,6,9,0.44),rgba(4,6,9,0.58)_34%,rgba(4,6,9,0.74)_62%,rgba(4,6,9,0.88))]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(240,208,150,0.1),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(113,154,212,0.1),transparent_26%)]" />
      </div>

      <div className="relative z-10 mx-auto flex max-w-7xl flex-col gap-8">
        {header}

        <main className="grid gap-6 rounded-[36px] border border-white/10 bg-[rgba(4,6,9,0.38)] px-4 py-6 backdrop-blur-[2px] sm:px-6 sm:py-8 pb-10">

          {/* Page header */}
          <div className="space-y-3">
            <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--accent-soft)]">
              {surface === "search" ? "Search" : "Feed"}
            </p>
            <h1 className="font-display text-5xl leading-none text-[var(--foreground)] sm:text-6xl">
              {surface === "search" ? "Search with context." : "Context-rich recommendations."}
            </h1>
          </div>

          {/* Guest notice */}
          {!isAuthenticated && (
            <div className="panel-glow mac-glass rounded-[26px] p-5">
              <p className="font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent-soft)] mb-2">
                Guest Mode
              </p>
              <p className="text-sm font-semibold text-[var(--foreground)]">
                {surface === "search"
                  ? "Search is live right away, even before sign-in."
                  : "The feed is live right away, even before sign-in."}
              </p>
              <p className="mt-1 text-sm leading-6 text-[var(--muted)]">
                Register or sign in to save your profile, search history, and long-term feedback.
              </p>
            </div>
          )}

          {/* Explanation banner */}
          {explanation && <ExplanationBanner text={explanation} mode={mode} />}

          {/* Skeleton loading */}
          {loading && articles.length === 0 && (
            <div className="grid gap-5 xl:grid-cols-3">
              {[...Array(6)].map((_, i) => (
                <div key={i} className="panel-glow mac-glass rounded-[28px] p-5 space-y-4">
                  <div className="h-4 w-20 animate-pulse rounded-full bg-white/[0.16]" />
                  <div className="h-8 w-3/4 animate-pulse rounded-full bg-white/[0.16]" />
                  <div className="h-4 w-full animate-pulse rounded-full bg-white/[0.14]" />
                  <div className="h-4 w-5/6 animate-pulse rounded-full bg-white/[0.14]" />
                </div>
              ))}
            </div>
          )}

          {/* Article grid */}
          {articles.length > 0 && (
            <>
              {loading && (
                <p className="text-sm text-[var(--muted)]">
                  {surface === "search" ? "Refreshing search results..." : "Refreshing feed..."}
                </p>
              )}

              <div className="grid gap-5 xl:grid-cols-3">
                {articles.map((article, index) => (
                  <div key={article.news_id} className="fade-in-up" style={{ animationDelay: `${index * 0.05}s` }}>
                    <NewsCard
                      article={article}
                      feedbackState={feedbackMap[normalizeArticleId(article.news_id)] || null}
                      onFeedback={(action) => void sendFeedback(article, index, action)}
                    />
                  </div>
                ))}
              </div>

              <div className="flex flex-col items-center gap-4 pt-2">
                {loadingMore && (
                  <p className="text-sm text-[var(--muted)]">Loading more stories...</p>
                )}
                {!loadingMore && hasMore && (
                  <button
                    onClick={loadMoreArticles}
                    className="inline-flex min-h-11 items-center justify-center rounded-full border border-white/16 bg-[linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0.06))] px-6 text-sm font-semibold tracking-[0.18em] uppercase text-[var(--foreground)] backdrop-blur-2xl transition hover:bg-[linear-gradient(180deg,rgba(255,255,255,0.2),rgba(255,255,255,0.08))]"
                  >
                    {surface === "search" ? "Load More Search Results" : "Load More News"}
                  </button>
                )}
                {!hasMore && (
                  <p className="text-sm text-[var(--muted)]">
                    {surface === "search"
                      ? "No more fresh search results in this batch."
                      : "No more fresh stories in this batch."}
                  </p>
                )}
                <div ref={loadMoreSentinelRef} className="h-px w-full" />
              </div>
            </>
          )}

          {/* Empty / welcome states */}
          {showWelcome && (
            <div className="grid place-items-center py-20 text-center space-y-4">
              <p className="font-display text-6xl text-[var(--foreground)]">Welcome.</p>
              <p className="text-sm leading-7 text-[var(--muted)] max-w-md">
                Your RL + KG recommendation feed is ready. Select a mood or wait for articles to load.
              </p>
            </div>
          )}

          {showSearchPrompt && (
            <div className="grid place-items-center py-20 text-center space-y-4">
              <p className="font-display text-6xl text-[var(--foreground)]">Search.</p>
              <p className="text-sm leading-7 text-[var(--muted)] max-w-md">
                Enter a query above to open a personalized search result feed.
              </p>
            </div>
          )}

          {showEmptyState && (
            <div className="grid place-items-center py-20 text-center space-y-4">
              <p className="font-display text-6xl text-[var(--foreground)]">
                {surface === "search" ? "No results." : "Empty."}
              </p>
              <p className="text-sm leading-7 text-[var(--muted)] max-w-md">
                {surface === "search"
                  ? "Try a different query or refresh to fetch a new batch."
                  : "Try refreshing for a fresh set or reset your profile memory."}
              </p>
            </div>
          )}
        </main>
      </div>

      {/* Consent modal */}
      {showConsentModal && (
        <div className="fixed inset-0 z-[10000] grid place-items-center bg-black/60 p-6 backdrop-blur-md">
          <div className="panel-glow mac-glass-heavy rounded-[28px] max-w-md w-full p-8 space-y-5">
            <div className="space-y-2">
              <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--accent-soft)]">Privacy</p>
              <h2 className="font-display text-4xl text-[var(--foreground)]">Enable Camera Mood?</h2>
            </div>
            <p className="text-sm leading-7 text-[var(--muted)]">
              HyperNews can read your facial expression via webcam to automatically detect your mood and personalise your feed in real time.
            </p>
            <ul className="space-y-2 text-sm text-[var(--muted)]">
              <li className="flex items-start gap-2">
                <span className="text-[var(--success)] mt-0.5">✓</span>
                All inference runs <strong className="text-[var(--foreground)]">on your device</strong> - no images or video are ever sent to a server.
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[var(--success)] mt-0.5">✓</span>
                Only the detected mood label (for example, happy) is used.
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[var(--success)] mt-0.5">✓</span>
                Turn it off anytime with the Camera On button.
              </li>
            </ul>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowConsentModal(false)}
                className="inline-flex min-h-10 items-center justify-center rounded-full border border-white/12 bg-[linear-gradient(180deg,rgba(255,255,255,0.09),rgba(255,255,255,0.03))] px-5 text-sm font-semibold tracking-[0.18em] uppercase text-[var(--muted-strong)] backdrop-blur-2xl transition"
              >
                Not Now
              </button>
              <button
                onClick={() => void handleConsentGrant()}
                className="inline-flex min-h-10 items-center justify-center rounded-full border border-[rgba(255,245,220,0.28)] bg-[linear-gradient(180deg,rgba(241,215,172,0.96),rgba(230,184,108,0.82))] px-5 text-sm font-semibold tracking-[0.18em] uppercase text-black shadow-[0_20px_60px_rgba(230,184,108,0.24)] transition"
              >
                Allow Camera
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {toastMsg && (
        <div className="fixed bottom-7 right-7 z-[9999] panel-glow mac-glass rounded-[20px] px-5 py-3 text-sm text-[var(--foreground)] fade-in-up">
          {toastMsg}
        </div>
      )}
    </div>
  );
}
