import OnboardingPageClient from "@/components/OnboardingPageClient";
import { requireIncompleteOnboardingProfile } from "@/lib/app-guard";

export default async function OnboardingPage() {
  const { profile } = await requireIncompleteOnboardingProfile();
  return <OnboardingPageClient initialProfile={profile} />;
}
