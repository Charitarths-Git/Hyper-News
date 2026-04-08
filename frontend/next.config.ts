import path from "path";
import type { NextConfig } from "next";

const monorepoRoot = path.resolve(__dirname, "..");

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  outputFileTracingRoot: monorepoRoot,
  turbopack: {
    root: monorepoRoot,
  },
};

export default nextConfig;
