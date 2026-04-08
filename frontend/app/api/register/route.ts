import { NextRequest } from "next/server";
import { proxyJsonRequest } from "@/lib/proxy";

export async function POST(request: NextRequest) {
  const payload = await request.json();
  return proxyJsonRequest("/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
