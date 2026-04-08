import RegisterPageClient from "@/components/RegisterPageClient";
import { redirectAuthenticatedUserAwayFromAuthPages } from "@/lib/app-guard";

export default async function RegisterPage() {
  await redirectAuthenticatedUserAwayFromAuthPages();
  return <RegisterPageClient />;
}
