/**
 * Persistent BYOK settings. The selected provider/model and the per-provider
 * API keys are stored in localStorage so they survive reloads — and ONLY in
 * localStorage. They are never sent to the retrieval API; rag.ts uses them to
 * call the provider directly from the browser.
 */
import { writable } from 'svelte/store';
import { browser } from '$app/environment';
import type { ProviderId } from './providers';

export interface ByokSettings {
	provider: ProviderId;
	/** Selected model id per provider (allows a custom value). */
	models: Partial<Record<ProviderId, string>>;
	/** API key per provider. */
	keys: Partial<Record<ProviderId, string>>;
}

const STORAGE_KEY = 'srs.byok.v1';

const DEFAULTS: ByokSettings = {
	provider: 'groq',
	models: {},
	keys: {}
};

function load(): ByokSettings {
	if (!browser) return DEFAULTS;
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (!raw) return DEFAULTS;
		return { ...DEFAULTS, ...JSON.parse(raw) };
	} catch {
		return DEFAULTS;
	}
}

function createByokStore() {
	const store = writable<ByokSettings>(load());
	if (browser) {
		store.subscribe((val) => {
			try {
				localStorage.setItem(STORAGE_KEY, JSON.stringify(val));
			} catch {
				/* storage full / disabled — ignore */
			}
		});
	}
	return store;
}

export const byok = createByokStore();

/** Wipe all stored keys (the "clear keys" button). */
export function clearKeys() {
	byok.update((s) => ({ ...s, keys: {} }));
}
