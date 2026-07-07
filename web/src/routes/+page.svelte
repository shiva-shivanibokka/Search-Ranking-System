<script lang="ts">
	import { onMount } from 'svelte';
	import { search, health, type SearchResponse, type HealthResponse } from '$lib/api';
	import { SAMPLE_QUERIES } from '$lib/samples';
	import ByokSettings from '$lib/components/ByokSettings.svelte';
	import StageBreakdown from '$lib/components/StageBreakdown.svelte';
	import ResultCard from '$lib/components/ResultCard.svelte';
	import RagAnswer from '$lib/components/RagAnswer.svelte';

	let query = $state('');
	let topK = $state(10);
	let ranker = $state<'lambdarank' | 'crossencoder'>('lambdarank');
	let useHyde = $state(true);

	let loading = $state(false);
	let error = $state('');
	let response = $state<SearchResponse | null>(null);
	let hstatus = $state<HealthResponse | null>(null);

	onMount(async () => {
		try {
			hstatus = await health();
		} catch {
			hstatus = null; // API not reachable yet (cold start / not deployed)
		}
	});

	async function run(q?: string) {
		const text = (q ?? query).trim();
		if (!text || loading) return;
		query = text;
		loading = true;
		error = '';
		try {
			response = await search({ query: text, top_k: topK, ranker, use_hyde: useHyde });
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			response = null;
		} finally {
			loading = false;
		}
	}

	function onKey(e: KeyboardEvent) {
		if (e.key === 'Enter') run();
	}
</script>

<svelte:head>
	<title>Neural Search Ranking System</title>
</svelte:head>

<header class="hero">
	<div class="container">
		<h1>Neural Search Ranking System</h1>
		<p class="sub">
			Two-stage neural search over ~1M MS&nbsp;MARCO passages — hybrid retrieval
			(FAISS&nbsp;+&nbsp;BM25 via Reciprocal Rank Fusion) then learned reranking, with
			optional client-side <strong>BYOK RAG</strong>.
		</p>
		<div class="status">
			{#if hstatus?.engine_ready}
				<span class="pill">🟢 engine ready</span>
				<span class="pill mono">{hstatus.index_size?.toLocaleString()} passages</span>
				<span class="pill mono">{hstatus.device}</span>
				{#if hstatus.llm_available}<span class="pill">HyDE: {hstatus.llm_provider}</span>{/if}
			{:else}
				<span class="pill">⚪ API not connected — set PUBLIC_API_URL (may be cold-starting)</span>
			{/if}
			<a class="pill" href="https://github.com/" target="_blank" rel="noreferrer">source ↗</a>
		</div>
	</div>
</header>

<main class="container grid">
	<section class="main">
		<div class="searchbar card">
			<input
				class="q"
				placeholder="Ask something factual, e.g. what causes inflation"
				bind:value={query}
				onkeydown={onKey}
			/>
			<button class="btn" onclick={() => run()} disabled={loading || !query.trim()}>
				{loading ? 'Searching…' : 'Search'}
			</button>
		</div>

		<div class="controls">
			<label>Top K
				<select bind:value={topK}>
					{#each [5, 10, 20] as k}<option value={k}>{k}</option>{/each}
				</select>
			</label>
			<label>Ranker
				<select bind:value={ranker}>
					<option value="lambdarank">LambdaRank</option>
					<option value="crossencoder">CrossEncoder</option>
				</select>
			</label>
			<label class="check">
				<input type="checkbox" bind:checked={useHyde} /> HyDE (if server key set)
			</label>
		</div>

		<div class="samples">
			<span class="dim">Try:</span>
			{#each SAMPLE_QUERIES as s (s)}
				<button class="sample" onclick={() => run(s)}>{s}</button>
			{/each}
		</div>

		{#if error}
			<div class="error card">⚠ {error}</div>
		{/if}

		{#if response}
			<RagAnswer query={response.query} passages={response.results} />
			<StageBreakdown data={response} />
			<div class="results card">
				<div class="rhead">
					<strong>Results</strong>
					<span class="dim">reranked by {response.ranker}</span>
				</div>
				{#each response.results as item, i (item.doc_id)}
					<ResultCard {item} citation={i < 5 ? i + 1 : undefined} />
				{/each}
			</div>
		{:else if !loading}
			<div class="empty card dim">
				Search the index, then generate a cited RAG answer with your own LLM key.
			</div>
		{/if}
	</section>

	<aside class="side">
		<ByokSettings />
		<div class="card explain">
			<strong>What you're looking at</strong>
			<ul>
				<li><b>Retrieval:</b> a DistilBERT two-tower encoder (FAISS IVF+PQ) and BM25 run in parallel, fused with RRF.</li>
				<li><b>Ranking:</b> a LambdaRank model reranks the fused candidates (CrossEncoder optional).</li>
				<li><b>RAG:</b> the answer is generated in <em>your</em> browser by <em>your</em> chosen LLM — the key never reaches the server.</li>
			</ul>
			<p class="dim honest">
				Honest limits: the dense retriever is trained on MS MARCO and generalizes only
				modestly out-of-domain; the demo searches a 1M-passage subset, not the full 8.8M
				collection. Numbers and tradeoffs are in the repo README.
			</p>
		</div>
	</aside>
</main>

<footer class="container">
	<p class="dim">
		Hybrid retrieval + learned ranking over MS MARCO · FastAPI on Cloud Run · SvelteKit on Vercel ·
		client-side BYOK RAG.
	</p>
</footer>

<style>
	.hero {
		border-bottom: 1px solid var(--border);
		background: linear-gradient(180deg, rgba(76, 141, 255, 0.08), transparent);
		padding: 34px 0 22px;
	}
	h1 {
		margin: 0 0 6px;
		font-size: 28px;
	}
	.sub {
		margin: 0 0 14px;
		color: var(--text-dim);
		max-width: 720px;
	}
	.status {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		align-items: center;
	}
	.grid {
		display: grid;
		grid-template-columns: 1fr 340px;
		gap: 20px;
		padding-top: 22px;
		padding-bottom: 22px;
		align-items: start;
	}
	.main {
		display: flex;
		flex-direction: column;
		gap: 14px;
		min-width: 0;
	}
	.side {
		display: flex;
		flex-direction: column;
		gap: 14px;
		position: sticky;
		top: 16px;
	}
	.searchbar {
		display: flex;
		gap: 10px;
		padding: 12px;
	}
	.q {
		flex: 1;
		background: var(--bg);
		border: 1px solid var(--border);
		color: var(--text);
		border-radius: 8px;
		padding: 12px 14px;
		font-size: 16px;
	}
	.controls {
		display: flex;
		gap: 16px;
		flex-wrap: wrap;
		align-items: center;
		font-size: 13px;
		color: var(--text-dim);
	}
	.controls label {
		display: inline-flex;
		gap: 6px;
		align-items: center;
	}
	.controls select {
		background: var(--surface);
		border: 1px solid var(--border);
		color: var(--text);
		border-radius: 6px;
		padding: 5px 8px;
	}
	.check {
		cursor: pointer;
	}
	.samples {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		align-items: center;
		font-size: 13px;
	}
	.sample {
		background: var(--surface);
		border: 1px solid var(--border);
		color: var(--text-dim);
		border-radius: 999px;
		padding: 5px 12px;
		font-size: 13px;
	}
	.sample:hover {
		color: var(--text);
		border-color: var(--accent);
	}
	.rhead,
	.results {
		display: flex;
		flex-direction: column;
	}
	.rhead {
		flex-direction: row;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 4px;
	}
	.empty {
		text-align: center;
		padding: 40px 20px;
	}
	.error {
		border-color: var(--warn);
		color: #f0c674;
	}
	.explain ul {
		margin: 8px 0;
		padding-left: 18px;
		font-size: 13px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.honest {
		margin: 6px 0 0;
		font-size: 12px;
	}
	footer {
		padding: 24px 20px 40px;
		font-size: 13px;
	}
	@media (max-width: 860px) {
		.grid {
			grid-template-columns: 1fr;
		}
		.side {
			position: static;
		}
	}
</style>
