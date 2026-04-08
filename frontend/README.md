# HyperNews Frontend

This is the Next.js frontend for HyperNews.

## Run Locally

```bash
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:3000
```

## Backend Requirement

The frontend expects the FastAPI backend to be running on:

```text
http://127.0.0.1:8000
```

If you do override `NEXT_PUBLIC_API_URL` on Windows, avoid adding a trailing space.

For cloud deployment on Vercel, set:

```env
AUTH_SECRET=<shared-secret>
NEXTAUTH_SECRET=<shared-secret>
NEXTAUTH_URL=https://<your-vercel-domain>
HYPERNEWS_BACKEND_URL=https://<your-render-domain>
NEXT_PUBLIC_API_URL=https://<your-render-domain>
```

## Notes

- `next.config.ts` pins Turbopack's root to the frontend directory to avoid workspace-root detection issues.
- If Turbopack becomes too heavy on your machine, you can use Webpack instead:

```bash
npx next dev --webpack
```

- The UI talks to:
  - `/recommend`
  - `/feedback`
  - `/graph`
  - `/profile/:user_id`
  - `/reset/:user_id`

- The recommended free deployment split is:
  - frontend on `Vercel Hobby`
  - backend on `Render` using the repo root `Dockerfile`
  - durable storage on `Supabase Postgres`
