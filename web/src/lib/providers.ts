/**
 * LLM provider + model registry for the client-side BYOK RAG panel.
 *
 * Every request in rag.ts is made *from the browser* straight to the provider
 * with the user's own key. Nothing here (keys, prompts, answers) touches the
 * retrieval API. Each provider lists a few common models plus a free-tier hint;
 * the UI also allows a custom model id, so this list never needs to be perfect.
 */

export type ProviderId = 'anthropic' | 'openai' | 'groq' | 'gemini';

export interface ModelOption {
	id: string;
	label: string;
	/** True for models with a usable free tier or very low cost. */
	free?: boolean;
}

export interface ProviderInfo {
	id: ProviderId;
	label: string;
	/** Where the user gets an API key. */
	keyUrl: string;
	/** Human hint about key format, shown under the input. */
	keyHint: string;
	models: ModelOption[];
}

export const PROVIDERS: Record<ProviderId, ProviderInfo> = {
	groq: {
		id: 'groq',
		label: 'Groq',
		keyUrl: 'https://console.groq.com/keys',
		keyHint: 'Free tier, very fast. Key starts with "gsk_".',
		models: [
			{ id: 'llama-3.3-70b-versatile', label: 'Llama 3.3 70B (versatile)', free: true },
			{ id: 'llama-3.1-8b-instant', label: 'Llama 3.1 8B (instant)', free: true },
			{ id: 'openai/gpt-oss-20b', label: 'GPT-OSS 20B', free: true }
		]
	},
	gemini: {
		id: 'gemini',
		label: 'Google Gemini',
		keyUrl: 'https://aistudio.google.com/apikey',
		keyHint: 'Free tier available. Key starts with "AIza".',
		models: [
			{ id: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash', free: true },
			{ id: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash', free: true },
			{ id: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' }
		]
	},
	openai: {
		id: 'openai',
		label: 'OpenAI',
		keyUrl: 'https://platform.openai.com/api-keys',
		keyHint: 'Paid. Key starts with "sk-".',
		models: [
			{ id: 'gpt-4o-mini', label: 'GPT-4o mini' },
			{ id: 'gpt-4o', label: 'GPT-4o' }
		]
	},
	anthropic: {
		id: 'anthropic',
		label: 'Anthropic',
		keyUrl: 'https://console.anthropic.com/settings/keys',
		keyHint: 'Paid. Key starts with "sk-ant-".',
		models: [
			{ id: 'claude-haiku-4-5', label: 'Claude Haiku 4.5' },
			{ id: 'claude-sonnet-5', label: 'Claude Sonnet 5' }
		]
	}
};

export const PROVIDER_ORDER: ProviderId[] = ['groq', 'gemini', 'openai', 'anthropic'];
