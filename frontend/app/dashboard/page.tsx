import DashboardPageClient from "@/components/DashboardPageClient";
import { requireCompletedOnboardingProfile } from "@/lib/app-guard";

export default async function DashboardPage() {
  const { profile } = await requireCompletedOnboardingProfile();
  return <DashboardPageClient initialProfile={profile} />;
}
