import { proxyJsonRequest } from "@/lib/proxy";

export async function GET() {
  return proxyJsonRequest("/graph");
}
