import RecommendationSurface from "@/components/RecommendationSurface";
import { redirectAuthenticatedUserToOnboardingIfNeeded } from "@/lib/app-guard";

export default async function FeedPage() {
  await redirectAuthenticatedUserToOnboardingIfNeeded();
  return <RecommendationSurface surface="feed" />;
}
