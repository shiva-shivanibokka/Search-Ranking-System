# Search Ranking — Web frontend (SvelteKit)

**Live:** https://web-shiv-a.vercel.app (auto-deployed from `main` via Vercel's Git integration; root directory `web`).

Client-side SPA for the Neural Search Ranking System. It calls the retrieval API
(`deploy/api.py` on Cloud Run) for hybrid retrieval + reranking, shows the full
pipeline stage breakdown, and generates a grounded **RAG answer** using the
visitor's own LLM key — entirely in the browser.

## Client-side BYOK RAG

The RAG panel supports **Groq, Google Gemini, OpenAI, and Anthropic** (with a
custom-model field for any of them). The API key is stored only in
`localStorage` and sent **directly from the browser to the provider** — it never
touches the retrieval server. See `src/lib/rag.ts` and `src/lib/providers.ts`.

## Local development

```bash
cd web
cp .env.example .env          # set PUBLIC_API_URL (default http://localhost:8080)
npm install
npm run dev                   # http://localhost:5173
```

Run the API locally in another terminal (from the repo root):

```bash
docker compose -f deploy/docker-compose.api.yml up --build
# or, with a local venv + artifacts present:
uvicorn deploy.api:app --port 8080
```

## Build

```bash
npm run build                 # static SPA in web/build/
npm run preview               # serve the production build
```

## Deploy to Vercel

1. Import the repo in Vercel and set **Root Directory** = `web`.
2. Framework preset: **SvelteKit** (auto-detected). Build command `npm run build`,
   output handled by the adapter.
3. Set env var `PUBLIC_API_URL` = your Cloud Run service URL.
4. After deploy, set `ALLOWED_ORIGINS` on the Cloud Run service to the Vercel URL
   so CORS only admits this frontend.

## Structure

| Path | Purpose |
|---|---|
| `src/lib/api.ts` | Typed client for the retrieval API (`/search`, `/health`). |
| `src/lib/providers.ts` | Provider + model registry (free/paid hints, key URLs). |
| `src/lib/rag.ts` | Browser-side RAG: builds a grounded prompt, calls the chosen provider. |
| `src/lib/stores.ts` | BYOK settings persisted to `localStorage`. |
| `src/lib/components/` | `ByokSettings`, `StageBreakdown`, `ResultCard`, `RagAnswer`. |
| `src/routes/+page.svelte` | The search + RAG page. |
