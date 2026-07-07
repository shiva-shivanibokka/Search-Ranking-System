/**
 * Client-side RAG. Given the passages the retrieval API returned, build a
 * grounded prompt and call the user's chosen LLM **directly from the browser**
 * with their own key. The key and prompt never touch our server.
 *
 * Each provider has a different endpoint/auth/body shape; `generateAnswer`
 * dispatches on provider id. All calls are CORS-friendly from the browser
 * (Anthropic requires an explicit opt-in header, included below).
 */
import type { ProviderId } from './providers';
import type { ResultItem } from './api';

export interface RagParams {
	provider: ProviderId;
	model: string;
	apiKey: string;
	query: string;
	passages: ResultItem[];
	maxPassages?: number;
}

const SYSTEM_PROMPT =
	'You are a precise search assistant. Answer the user question using ONLY the ' +
	'numbered passages provided. Cite the passages you use inline like [1], [2]. ' +
	'If the passages do not contain the answer, say so plainly instead of guessing. ' +
	'Answer in 2–5 sentences and stop once the question is answered.';

function buildUserPrompt(query: string, passages: ResultItem[], maxPassages: number): string {
	const ctx = passages
		.slice(0, maxPassages)
		.map((p, i) => `[${i + 1}] (doc ${p.doc_id}) ${p.text}`)
		.join('\n\n');
	return `Question: ${query}\n\nPassages:\n${ctx}\n\nAnswer (with [n] citations):`;
}

async function readError(res: Response): Promise<string> {
	try {
		const body = await res.json();
		return body?.error?.message || body?.error || body?.message || JSON.stringify(body);
	} catch {
		return `${res.status} ${res.statusText}`;
	}
}

/** OpenAI + Groq share the OpenAI chat-completions schema. */
async function openaiCompatible(
	baseUrl: string,
	params: RagParams,
	maxPassages: number
): Promise<string> {
	const res = await fetch(`${baseUrl}/chat/completions`, {
		method: 'POST',
		headers: {
			'content-type': 'application/json',
			authorization: `Bearer ${params.apiKey}`
		},
		body: JSON.stringify({
			model: params.model,
			temperature: 0.2,
			max_tokens: 600,
			messages: [
				{ role: 'system', content: SYSTEM_PROMPT },
				{ role: 'user', content: buildUserPrompt(params.query, params.passages, maxPassages) }
			]
		})
	});
	if (!res.ok) throw new Error(await readError(res));
	const data = await res.json();
	return data?.choices?.[0]?.message?.content?.trim() ?? '(empty response)';
}

async function anthropic(params: RagParams, maxPassages: number): Promise<string> {
	const res = await fetch('https://api.anthropic.com/v1/messages', {
		method: 'POST',
		headers: {
			'content-type': 'application/json',
			'x-api-key': params.apiKey,
			'anthropic-version': '2023-06-01',
			// Opt in to browser-origin calls (Anthropic blocks them by default).
			'anthropic-dangerous-direct-browser-access': 'true'
		},
		body: JSON.stringify({
			model: params.model,
			max_tokens: 600,
			temperature: 0.2,
			system: SYSTEM_PROMPT,
			messages: [
				{ role: 'user', content: buildUserPrompt(params.query, params.passages, maxPassages) }
			]
		})
	});
	if (!res.ok) throw new Error(await readError(res));
	const data = await res.json();
	const text = (data?.content ?? [])
		.filter((b: { type: string }) => b.type === 'text')
		.map((b: { text: string }) => b.text)
		.join('');
	return text.trim() || '(empty response)';
}

async function gemini(params: RagParams, maxPassages: number): Promise<string> {
	const url =
		`https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(params.model)}` +
		`:generateContent?key=${encodeURIComponent(params.apiKey)}`;
	const res = await fetch(url, {
		method: 'POST',
		headers: { 'content-type': 'application/json' },
		body: JSON.stringify({
			system_instruction: { parts: [{ text: SYSTEM_PROMPT }] },
			contents: [
				{
					role: 'user',
					parts: [{ text: buildUserPrompt(params.query, params.passages, maxPassages) }]
				}
			],
			generationConfig: { temperature: 0.2, maxOutputTokens: 600 }
		})
	});
	if (!res.ok) throw new Error(await readError(res));
	const data = await res.json();
	const text = (data?.candidates?.[0]?.content?.parts ?? [])
		.map((p: { text?: string }) => p.text ?? '')
		.join('');
	return text.trim() || '(empty response)';
}

export async function generateAnswer(params: RagParams): Promise<string> {
	const maxPassages = params.maxPassages ?? 5;
	if (!params.apiKey) throw new Error('No API key set for this provider.');
	switch (params.provider) {
		case 'groq':
			return openaiCompatible('https://api.groq.com/openai/v1', params, maxPassages);
		case 'openai':
			return openaiCompatible('https://api.openai.com/v1', params, maxPassages);
		case 'anthropic':
			return anthropic(params, maxPassages);
		case 'gemini':
			return gemini(params, maxPassages);
		default:
			throw new Error(`Unknown provider: ${params.provider}`);
	}
}
