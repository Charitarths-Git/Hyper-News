# Free Cloud Deployment

This repo now supports the recommended free deployment split:

- `Vercel Hobby` for the Next.js frontend
- `Render Free Web Service` for the FastAPI backend using the repo `Dockerfile`
- `Supabase Free Postgres` for durable storage

`docker-compose.yml` remains a local/full-stack reference. It is not the cloud deployment artifact for the free setup.

## 1. Prepare a deploy branch

Create a clean deploy branch and build the slim runtime asset set there:

```powershell
git checkout -b deploy/free-cloud
python .\scripts\prepare_cloud_assets.py --limit 2000 --in-place
```

That keeps the backend runtime artifact small enough for free hosting while preserving category diversity.

Tracked deploy assets should be limited to:

- `data/articles.parquet`
- `data/article_embeddings.npy`
- `models/ltr_model.txt`
- `models/ltr_model.txt.weights.json`

Everything else under `data/`, `graph/`, and bulky local outputs stays ignored.

## 2. Configure Supabase

Create a free Supabase project and copy a PostgreSQL connection string that includes `sslmode=require`.

Then migrate your current local SQLite state:

```powershell
$env:DATABASE_URL="postgresql://<user>:<password>@<host>:5432/postgres?sslmode=require"
python .\backend\migrate_sqlite_to_postgres.py --sqlite-path .\data\hypernews.db
```

The backend now forwards supported PostgreSQL query parameters from `DATABASE_URL`, so `sslmode=require` works for hosted Postgres providers.

## 3. Deploy the backend on Render

Create a new Render Web Service from the repo and choose Docker.

Use the repo `Dockerfile` at the project root.

Set the health check path to:

```text
/health
```

Set these environment variables:

```env
DATABASE_URL=postgresql://<user>:<password>@<host>:5432/postgres?sslmode=require
AUTH_SECRET=<shared-secret>
HYPERNEWS_MAX_ARTICLES=2000
HYPERNEWS_GRAPH_ARTICLE_LIMIT=1000
HYPERNEWS_ENABLE_GRAPH=1
HYPERNEWS_ENABLE_SENTENCE_TRANSFORMER=0
HYPERNEWS_ENABLE_RERANKER=0
HYPERNEWS_ENABLE_QDRANT=0
HYPERNEWS_ENABLE_OPENSEARCH=0
```

Leave these unset in the free deployment:

- `REDIS_URL`
- `QDRANT_URL`
- `OPENSEARCH_URL`

Optional local verification if Docker is available on your machine:

```powershell
docker build -t hypernews-backend .
docker run --rm -p 8000:8000 --env-file .env hypernews-backend
```

## 4. Deploy the frontend on Vercel

Deploy the `frontend/` directory as a Vercel project.

Set these environment variables:

```env
AUTH_SECRET=<shared-secret>
NEXTAUTH_SECRET=<shared-secret>
NEXTAUTH_URL=https://<your-vercel-domain>
HYPERNEWS_BACKEND_URL=https://<your-render-domain>
NEXT_PUBLIC_API_URL=https://<your-render-domain>
```

The frontend remains non-Docker on purpose. Vercel does not run Docker images directly, so Docker is only used where it materially helps: the backend.

## 5. Smoke test

After both deployments are live:

1. Open the Vercel frontend URL.
2. Confirm the Render backend returns `200` from `/health`.
3. Register a new account and sign in.
4. Confirm feed recommendations load.
5. Confirm search works in slim cloud mode.
6. Confirm feedback persists after a backend restart/redeploy.

## Fallback profile

If the Render free backend OOMs or fails to boot reliably, redeploy with:

```env
HYPERNEWS_ENABLE_GRAPH=0
HYPERNEWS_MAX_ARTICLES=1000
HYPERNEWS_GRAPH_ARTICLE_LIMIT=0
```
