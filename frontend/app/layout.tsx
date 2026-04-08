import type { Metadata } from "next";
import { Manrope } from "next/font/google";
import { auth } from "@/auth";
import AuthProvider from "@/components/AuthProvider";
import "./globals.css";

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "HyperNews - AI-Powered Personalized News",
  description:
    "Hyper-personalized news recommendations powered by RAG, knowledge graphs, and reinforcement learning.",
  openGraph: {
    title: "HyperNews",
    description: "Your mood-aware, context-intelligent news feed.",
  },
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();
  return (
    <html lang="en" className={`${manrope.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <AuthProvider session={session}>{children}</AuthProvider>
      </body>
    </html>
  );
}
