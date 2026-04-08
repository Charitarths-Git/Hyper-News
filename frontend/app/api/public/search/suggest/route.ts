import { NextRequest } from "next/server";
import { proxyJsonRequest } from "@/lib/proxy";

function normalizeGuestId(value: string | null): string {
  const guestId = String(value || "").trim();
  return guestId || "guest_demo";
}

export async function GET(request: NextRequest) {
  const url = new URL(request.url);
  const query = url.searchParams.get("q") || "";
  const limit = url.searchParams.get("limit") || "10";
  const guestId = normalizeGuestId(url.searchParams.get("guest_id"));

  return proxyJsonRequest(
    `/search/suggest?q=${encodeURIComponent(query)}&user_id=${encodeURIComponent(guestId)}&limit=${encodeURIComponent(limit)}`,
  );
}
