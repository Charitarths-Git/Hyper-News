import { NextRequest } from "next/server";
import { proxyJsonRequest, requireSessionUser } from "@/lib/proxy";

export async function GET(request: NextRequest) {
  const { userId, response } = await requireSessionUser();
  if (response) {
    return response;
  }

  const url = new URL(request.url);
  const query = url.searchParams.get("q") || "";
  const limit = url.searchParams.get("limit") || "10";
  return proxyJsonRequest(
    `/search/suggest?q=${encodeURIComponent(query)}&user_id=${encodeURIComponent(userId)}&limit=${encodeURIComponent(limit)}`,
  );
}
