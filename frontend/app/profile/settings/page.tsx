import ProfileSettingsPage from "@/components/ProfileSettingsPage";
import { requireCompletedOnboardingProfile } from "@/lib/app-guard";

export default async function ProfileSettingsRoute() {
  const { profile } = await requireCompletedOnboardingProfile();
  return <ProfileSettingsPage initialProfile={profile} />;
}
