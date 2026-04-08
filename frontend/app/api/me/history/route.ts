import { NextRequest } from "next/server";
import { proxyJsonRequest, requireSessionUser } from "@/lib/proxy";

export async function GET(request: NextRequest) {
  const { userId, response } = await requireSessionUser();
  if (response) {
    return response;
  }
  const url = new URL(request.url);
  const limit = url.searchParams.get("limit") || "25";
  return proxyJsonRequest(`/me/history?user_id=${encodeURIComponent(userId)}&limit=${encodeURIComponent(limit)}`);
}
