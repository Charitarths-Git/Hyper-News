import { NextRequest, NextResponse } from "next/server";
import { proxyJsonRequest } from "@/lib/proxy";

function normalizeGuestId(value: unknown): string {
  const guestId = String(value || "").trim();
  return guestId || "";
}

export async function POST(request: NextRequest) {
  const payload = await request.json().catch(() => ({}));
  const guestId = normalizeGuestId(payload?.guest_id);
  if (!guestId) {
    return NextResponse.json({ error: "guest_id is required" }, { status: 400 });
  }

  return proxyJsonRequest(`/reset/${encodeURIComponent(guestId)}`, {
    method: "POST",
  });
}
