/**
 * Client for the retrieval API (deploy/api.py). Talks to PUBLIC_API_URL, which
 * is the Cloud Run service in production or a local uvicorn in dev.
 */
import { env } from '$env/dynamic/public';

const BASE = (env.PUBLIC_API_URL || 'http://localhost:8080').replace(/\/$/, '');

// Local dev uses the localhost default silently. But if a *deployed* build (not
// on localhost) fell back to localhost, PUBLIC_API_URL was never set at build —
// fail loudly with a clear message instead of firing every request at localhost.
function assertConfigured(): void {
	if (typeof window === 'undefined') return;
	const onLocalhost = /^(localhost|127\.|0\.0\.0\.0)/.test(window.location.hostname);
	const baseLocal = /localhost|127\.0\.0\.1/.test(BASE);
	if (baseLocal && !onLocalhost) {
		throw new Error(
			'Search API is not configured for this deployment: PUBLIC_API_URL was unset at ' +
				'build time, so the app is pointing at localhost. Set PUBLIC_API_URL in the Vercel project.'
		);
	}
}

export interface StageCandidate {
	doc_id: number;
	score: number;
	rank: number;
}

export interface ResultItem {
	rank: number;
	doc_id: number;
	text: string;
	score: number;
	ranker: string;
}

export interface SearchResponse {
	request_id: string;
	query: string;
	ranker: string;
	results: ResultItem[];
	stages: {
		intent: string;
		hyde_used: boolean;
		embed_text_preview: string;
		dense_top: StageCandidate[];
		sparse_top: StageCandidate[];
		fused_count: number;
	};
	timings: {
		hyde_ms: number;
		retrieve_ms: number;
		rerank_ms: number;
		total_ms: number;
	};
}

export interface SearchParams {
	query: string;
	top_k?: number;
	candidates?: number;
	ranker?: 'lambdarank' | 'crossencoder';
	use_hyde?: boolean;
}

export async function search(params: SearchParams): Promise<SearchResponse> {
	assertConfigured();
	const res = await fetch(`${BASE}/search`, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({ top_k: 10, ...params })
	});
	if (res.status === 429) {
		throw new Error('Rate limit reached — please wait a moment and try again.');
	}
	if (res.status === 503) {
		throw new Error('The search engine is still warming up (cold start). Retry in ~30s.');
	}
	if (!res.ok) {
		let detail = `${res.status} ${res.statusText}`;
		try {
			const body = await res.json();
			if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail;
		} catch {
			/* ignore */
		}
		throw new Error(`Search failed: ${detail}`);
	}
	return (await res.json()) as SearchResponse;
}

export interface HealthResponse {
	status: string;
	engine_ready: boolean;
	index_size?: number;
	device?: string;
	cross_encoder?: boolean;
	llm_provider?: string;
	llm_available?: boolean;
}

export async function health(): Promise<HealthResponse> {
	assertConfigured();
	const res = await fetch(`${BASE}/health`);
	if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
	return (await res.json()) as HealthResponse;
}

export const API_BASE = BASE;
