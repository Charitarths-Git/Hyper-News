import { NextRequest } from "next/server";
import { proxyJsonRequest, requireSessionUser } from "@/lib/proxy";

export async function POST(request: NextRequest) {
  const { userId, response } = await requireSessionUser();
  if (response) {
    return response;
  }

  const payload = await request.json();
  return proxyJsonRequest("/recommend", {
    method: "POST",
    body: JSON.stringify({
      ...payload,
      user_id: userId,
      session_id: payload?.session_id || userId,
    }),
  });
}
