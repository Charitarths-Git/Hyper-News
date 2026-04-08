import RecommendationSurface from "@/components/RecommendationSurface";
import { redirectAuthenticatedUserToOnboardingIfNeeded } from "@/lib/app-guard";

type SearchParams = {
  q?: string | string[];
};

export default async function SearchPage({
  searchParams,
}: {
  searchParams?: Promise<SearchParams>;
}) {
  await redirectAuthenticatedUserToOnboardingIfNeeded();

  const resolvedSearchParams = await searchParams;
  const queryValue = resolvedSearchParams?.q;
  const query = Array.isArray(queryValue) ? queryValue[0] || "" : queryValue || "";

  return <RecommendationSurface key={`search-${query}`} surface="search" initialQuery={query} />;
}
