# 0007 — Cloud Run retrieval API + SvelteKit/Vercel frontend with client-side BYOK RAG

**Status:** Accepted (supersedes the Gradio/HF-Space parts of [ADR 0004](0004-free-tier-deployment-topology.md))

## Context
ADR 0004 put the consolidated demo behind a Gradio UI on a free Hugging Face
Space. Two things forced a change:
1. **The free tiers were exhausted** — Render, Hugging Face Spaces, and Supabase
   were all used up on this account, so the Space path was no longer available.
2. **Portfolio bar** — a Gradio UI reads as a notebook demo. The project needs a
   production-grade frontend, and a retrieval-augmented-generation (RAG) story on
   top of the ranking system, without incurring server-side LLM cost.

The backend is ~1.2 GB of artifacts and needs ~2–4 GB RAM, which does not fit a
no-credit-card free tier.

## Decision
- **Retrieval API:** extract a FastAPI service (`deploy/api.py`) over the existing
  consolidated engine, exposing `/search` (with a full pipeline stage breakdown)
  and `/health`. Host it on **Google Cloud Run**: scale-to-zero (idle ≈ $0), free
  monthly allowance covers a demo, and it can be sized to 4 GB. A Cloud Scheduler
  ping to `/health` mitigates cold starts.
- **Frontend:** a **SvelteKit** SPA (`web/`, adapter-static) on **Vercel**. It
  renders the search UI, the stage breakdown, and the RAG panel.
- **RAG:** **client-side, bring-your-own-key**. The visitor picks a provider
  (Groq / Gemini / OpenAI / Anthropic) and pastes their own key, which is stored
  only in the browser and sent straight to the provider from the browser
  (`web/src/lib/rag.ts`). The retrieval server never sees the key or the prompt.

## Consequences
- **Pro:** a real, clickable, production-shaped demo at ≈ $0; hosting cost is
  bounded by scale-to-zero, and there is zero server-side LLM spend.
- **Pro:** the key-never-touches-server design is an honest, defensible security
  posture (and a good interview talking point).
- **Pro:** the API is reusable — the same `/search` contract could back any client.
- **Con:** a Cloud Run cold start is ~60–120 s (artifact download + model load);
  the keep-warm ping helps but does not fully eliminate it on a free budget.
- **Con:** browser-origin LLM calls depend on each provider allowing CORS
  (Anthropic needs an explicit opt-in header); a provider could change this.
- **Con:** Cloud Run requires a card on file even though usage stays in the free
  tier — acceptable, and cheaper than any always-on host.

## Alternatives considered
- **Keep Gradio on HF Spaces** — free tier exhausted; also the UI we wanted to move off.
- **Fly.io** — viable scale-to-zero alternative, but the user could not use it.
- **Server-side RAG with a shared key** — real LLM cost on a public demo and a
  secret to protect; rejected in favor of BYOK.
- **Shrink to a dense-only index for a lighter host** — would drop the hybrid
  (BM25 + RRF) story, which is a core part of the system; rejected.
