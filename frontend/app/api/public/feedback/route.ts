import { NextRequest } from "next/server";
import { proxyJsonRequest } from "@/lib/proxy";

function normalizeGuestId(value: unknown): string {
  const guestId = String(value || "").trim();
  return guestId || "guest_demo";
}

export async function POST(request: NextRequest) {
  const payload = await request.json();
  const guestId = normalizeGuestId(payload?.guest_id);

  return proxyJsonRequest("/feedback", {
    method: "POST",
    body: JSON.stringify({
      ...payload,
      guest_id: undefined,
      user_id: guestId,
      session_id: payload?.session_id || guestId,
    }),
  });
}
