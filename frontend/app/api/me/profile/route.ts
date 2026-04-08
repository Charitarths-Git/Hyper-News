import { proxyJsonRequest, requireSessionUser } from "@/lib/proxy";

export async function GET() {
  const { userId, response } = await requireSessionUser();
  if (response) {
    return response;
  }
  return proxyJsonRequest(`/me/profile?user_id=${encodeURIComponent(userId)}`);
}

export async function POST(request: Request) {
  const { userId, response } = await requireSessionUser();
  if (response) {
    return response;
  }
  const payload = await request.json();
  return proxyJsonRequest(`/me/profile?user_id=${encodeURIComponent(userId)}`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
