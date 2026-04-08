import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";
import { serverApiUrl } from "@/lib/server-api";

const authSecret =
  process.env.AUTH_SECRET ||
  process.env.NEXTAUTH_SECRET ||
  (process.env.NODE_ENV !== "production"
    ? "hypernews-local-dev-secret-change-me"
    : undefined);

export const { handlers, auth, signIn, signOut } = NextAuth({
  secret: authSecret,
  trustHost: true,
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },
  providers: [
    Credentials({
      name: "HyperNews Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        const email = String(credentials?.email || "").trim();
        const password = String(credentials?.password || "");

        if (!email || !password) {
          return null;
        }

        const response = await fetch(serverApiUrl("/auth/validate"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
          cache: "no-store",
        });

        if (!response.ok) {
          return null;
        }

        const payload = await response.json();
        const user = payload?.user;
        if (!user?.user_id) {
          return null;
        }

        return {
          id: String(user.user_id),
          email: String(user.email || email),
          name: String(user.display_name || email.split("@")[0]),
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.sub = user.id;
        token.name = user.name;
        token.email = user.email;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = String(token.sub || "");
        session.user.name = String(token.name || session.user.name || "");
        session.user.email = String(token.email || session.user.email || "");
      }
      return session;
    },
  },
});
