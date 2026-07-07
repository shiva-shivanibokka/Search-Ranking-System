import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		// Static SPA: everything runs client-side (the BYOK RAG calls the user's
		// LLM directly from the browser). `fallback` makes it a single-page app.
		adapter: adapter({ fallback: 'index.html' })
	}
};

export default config;
