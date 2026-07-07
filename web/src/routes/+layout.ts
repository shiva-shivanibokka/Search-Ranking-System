// Pure client-side SPA: no server rendering (the BYOK RAG and all state live in
// the browser). adapter-static emits an index.html fallback that boots the app.
export const ssr = false;
export const prerender = false;
