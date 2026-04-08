import { proxyJsonRequest, requireSessionUser } from "@/lib/proxy";

export async function POST() {
  const { userId, response } = await requireSessionUser();
  if (response) {
    return response;
  }
  return proxyJsonRequest(`/reset/${encodeURIComponent(userId)}`, {
    method: "POST",
  });
}
