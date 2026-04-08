import LoginPageClient from "@/components/LoginPageClient";
import { redirectAuthenticatedUserAwayFromAuthPages } from "@/lib/app-guard";

export default async function LoginPage() {
  await redirectAuthenticatedUserAwayFromAuthPages();
  return <LoginPageClient />;
}
