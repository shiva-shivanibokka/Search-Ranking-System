<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { search, health, type SearchResponse, type HealthResponse } from '$lib/api';
	import { SAMPLE_QUERIES } from '$lib/samples';
	import PipelineRail from '$lib/components/PipelineRail.svelte';
	import RagBox from '$lib/components/RagBox.svelte';
	import ResultCard from '$lib/components/ResultCard.svelte';

	let query = $state('');
	let topK = $state(10);
	let ranker = $state<'lambdarank' | 'crossencoder'>('lambdarank');
	let useHyde = $state(true);

	let loading = $state(false);
	let error = $state('');
	let response = $state<SearchResponse | null>(null);
	let hstatus = $state<HealthResponse | null>(null);

	// Which pipeline stage is currently lit (0=Understand … 3=Answer, -1=idle).
	// The search is one API call, so we walk the highlight through the stages while
	// it runs, then land on Answer (where the user drops in their key).
	let activeStage = $state(-1);
	let seqTimer: ReturnType<typeof setInterval> | null = null;

	function clearSeq() {
		if (seqTimer) {
			clearInterval(seqTimer);
			seqTimer = null;
		}
	}

	onMount(async () => {
		try {
			hstatus = await health();
		} catch {
			hstatus = null;
		}
	});
	onDestroy(clearSeq);

	async function run(q?: string) {
		const text = (q ?? query).trim();
		if (!text || loading) return;
		query = text;
		loading = true;
		error = '';
		response = null; // clear so the pipeline rail shows its running state

		// Walk the highlight Understand -> Retrieve -> Rank while the request is in
		// flight; hold on Rank (the slow stage) until the response arrives.
		clearSeq();
		activeStage = 0;
		let step = 0;
		seqTimer = setInterval(() => {
			if (step < 2) activeStage = ++step;
		}, 750);

		try {
			response = await search({ query: text, top_k: topK, ranker, use_hyde: useHyde });
			activeStage = 3; // land on Answer — invite the key
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
			response = null;
			activeStage = -1;
		} finally {
			loading = false;
			clearSeq();
		}
	}
</script>

<svelte:head><title>Neural Search Ranking System</title></svelte:head>

<header class="topbar">
	<div class="wrap bar">
		<div class="brand">
			<span class="glyph" aria-hidden="true"></span>
			<span class="word">Neural&nbsp;Search<span class="accent">·</span>Ranking</span>
		</div>
		<div class="status">
			{#if hstatus?.engine_ready}
				<span class="pill live"><span class="dot"></span> engine ready</span>
				<span class="pill mono">{hstatus.index_size?.toLocaleString()} passages</span>
				<span class="pill mono">{hstatus.device}</span>
			{:else}
				<span class="pill"><span class="dot cold"></span> API waking up — first call cold-starts (~1–2 min)</span>
			{/if}
			<a class="pill" href="https://github.com/shiva-shivanibokka/Search-Ranking-System" target="_blank" rel="noreferrer">GitHub ↗</a>
		</div>
	</div>
</header>

<!-- Explainer strip, up top -->
<div class="wrap">
	<div class="explain">
		<div class="ex ex-dense">
			<span class="k dense-k">Retrieve</span>
			<span class="v">A DistilBERT two-tower (<span class="dense">FAISS</span>) and <span class="sparse">BM25</span> search in parallel, fused with Reciprocal Rank Fusion.</span>
		</div>
		<div class="ex ex-rank">
			<span class="k rank-k">Rank</span>
			<span class="v">A LambdaRank model reranks the fused candidates over a <b>1M-passage</b> index (Recall@100 ≈ 0.74).</span>
		</div>
		<div class="ex ex-rag">
			<span class="k rag-k">Answer</span>
			<span class="v">Optional RAG runs in <b>your</b> browser with <b>your</b> LLM key — it never touches the server.</span>
		</div>
	</div>
</div>

<main class="wrap page">
	<!-- Search -->
	<section class="search card">
		<div class="searchrow">
			<span class="mag" aria-hidden="true">⌕</span>
			<input
				class="q"
				placeholder="Ask something factual — e.g. what causes inflation"
				bind:value={query}
				onkeydown={(e) => e.key === 'Enter' && run()}
			/>
			<button class="btn" onclick={() => run()} disabled={loading || !query.trim()}>
				{loading ? 'Searching…' : 'Search'}
			</button>
		</div>
		<div class="controls">
			<label>Top&nbsp;K
				<select bind:value={topK}>{#each [5, 10, 20] as k}<option value={k}>{k}</option>{/each}</select>
			</label>
			<label>Ranker
				<select bind:value={ranker}>
					<option value="lambdarank">LambdaRank</option>
					<option value="crossencoder">CrossEncoder</option>
				</select>
			</label>
			<label class="chk"><input type="checkbox" bind:checked={useHyde} /> HyDE</label>
			<span class="try dim">try</span>
			<div class="chips">
				{#each SAMPLE_QUERIES as sq (sq)}
					<button class="chip" onclick={() => run(sq)}>{sq}</button>
				{/each}
			</div>
		</div>
	</section>

	{#if error}
		<div class="banner err">⚠ {error}</div>
	{/if}

	{#if loading || response}
		<PipelineRail data={response} active={activeStage} {loading} />
	{/if}

	{#if response}
		<RagBox query={response.query} passages={response.results} />

		<!-- Dense + Sparse side by side, full width -->
		<div class="cand-row">
			<section class="card col dense-col">
				<div class="secttl">
					<span class="dense">Dense · FAISS</span>
					<span class="dim mono">top {topK} · cosine</span>
				</div>
				<ol class="clist">
					{#each response.stages.dense_top as c (c.doc_id)}
						<li><span class="r mono">{c.rank}</span><span class="d mono">doc {c.doc_id}</span><span class="s mono dense">{c.score.toFixed(3)}</span></li>
					{/each}
				</ol>
			</section>
			<section class="card col sparse-col">
				<div class="secttl">
					<span class="sparse">Sparse · BM25</span>
					<span class="dim mono">top {topK} · bm25</span>
				</div>
				<ol class="clist">
					{#each response.stages.sparse_top as c (c.doc_id)}
						<li><span class="r mono">{c.rank}</span><span class="d mono">doc {c.doc_id}</span><span class="s mono sparse">{c.score.toFixed(1)}</span></li>
					{/each}
				</ol>
			</section>
		</div>

		<!-- Reranked results, full width underneath -->
		<section class="results card">
			<div class="secttl">
				<span>Results</span>
				<span class="dim mono">reranked · {response.ranker}</span>
			</div>
			{#each response.results as item, i (item.doc_id)}
				<ResultCard {item} citation={i < 5 ? i + 1 : undefined} />
			{/each}
		</section>
	{:else if !loading}
		<div class="empty card">
			<div class="big">Watch a real retrieval pipeline work.</div>
			<p class="dim">Type a query or pick one above — you'll see the dense and sparse candidates, the fusion, the rerank, and the timing of every stage. Bring your own LLM key to generate a cited answer.</p>
		</div>
	{/if}

	<footer class="foot dim">
		<span>Hybrid retrieval + learned ranking over MS MARCO · FastAPI on Cloud&nbsp;Run · SvelteKit on Vercel · client-side BYOK RAG.</span>
		<span class="lim">Free-tier demo: dense retriever generalises only modestly out-of-domain; ~1M-passage subset, not the full 8.8M. Numbers &amp; tradeoffs in the repo README.</span>
	</footer>
</main>

<style>
	/* ── top bar ── */
	.topbar {
		border-bottom: 1px solid var(--border-soft);
		background: rgba(10, 14, 28, 0.6);
		backdrop-filter: blur(8px);
		position: sticky;
		top: 0;
		z-index: 10;
	}
	.bar {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 16px;
		padding-top: 14px;
		padding-bottom: 14px;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
		font-weight: 700;
		font-size: 17px;
		letter-spacing: -0.02em;
	}
	.glyph {
		width: 20px;
		height: 20px;
		border-radius: 6px;
		background: conic-gradient(from 140deg, var(--dense), var(--primary), var(--sparse), var(--dense));
		box-shadow: 0 0 18px -4px var(--primary);
	}
	.word .accent {
		color: var(--primary-2);
		margin: 0 2px;
	}
	.status {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		align-items: center;
	}
	.pill.live {
		color: var(--good);
		border-color: color-mix(in srgb, var(--good) 40%, var(--border));
	}
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--good);
		box-shadow: 0 0 8px var(--good);
	}
	.dot.cold {
		background: var(--sparse);
		box-shadow: 0 0 8px var(--sparse);
	}

	/* ── explainer strip ── */
	.explain {
		display: grid;
		grid-template-columns: repeat(3, 1fr);
		gap: 14px;
		margin-top: 22px;
	}
	.ex {
		display: flex;
		flex-direction: column;
		gap: 6px;
		padding: 14px 16px 14px 18px;
		background: linear-gradient(180deg, color-mix(in srgb, var(--ac, var(--primary)) 8%, var(--surface)), transparent);
		border: 1px solid color-mix(in srgb, var(--ac, var(--border)) 35%, var(--border-soft));
		border-left: 3px solid var(--ac, var(--primary));
		border-radius: var(--radius-sm);
	}
	.ex-dense {
		--ac: var(--dense);
	}
	.ex-rank {
		--ac: var(--primary-2);
	}
	.ex-rag {
		--ac: var(--good);
	}
	.k {
		font-size: 11px;
		font-weight: 700;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		width: fit-content;
	}
	.dense-k {
		color: var(--dense);
	}
	.rank-k {
		color: var(--primary-2);
	}
	.rag-k {
		color: var(--good);
	}
	.ex .v {
		font-size: 13px;
		color: var(--text-dim);
		line-height: 1.5;
	}
	.dense {
		color: var(--dense);
		font-weight: 600;
	}
	.sparse {
		color: var(--sparse);
		font-weight: 600;
	}

	/* ── page ── */
	.page {
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding-top: 16px;
		padding-bottom: 40px;
	}

	/* ── search ── */
	.search {
		padding: 16px 18px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		border-top: 2px solid color-mix(in srgb, var(--primary) 55%, var(--border));
		box-shadow: 0 0 32px -20px var(--primary);
	}
	.searchrow {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.mag {
		font-size: 22px;
		color: var(--text-faint);
		padding-left: 4px;
	}
	.q {
		flex: 1;
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 14px 16px;
		font-size: 17px;
	}
	.q:focus {
		border-color: var(--primary);
		outline: none;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
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
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 6px 9px;
	}
	.chk {
		cursor: pointer;
	}
	.try {
		margin-left: 6px;
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
	}
	.chips {
		display: flex;
		gap: 7px;
		flex-wrap: wrap;
	}
	.chip {
		background: var(--surface-2);
		border: 1px solid var(--border);
		color: var(--text-dim);
		border-radius: 999px;
		padding: 5px 12px;
		font-size: 12.5px;
		transition: all 0.14s ease;
	}
	.chip:hover {
		color: var(--text);
		border-color: var(--dense);
		box-shadow: 0 0 0 1px color-mix(in srgb, var(--dense) 40%, transparent);
	}

	/* ── dense/sparse candidates: side-by-side row, full width ── */
	.cand-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}
	.secttl {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 10px;
		margin-bottom: 8px;
		font-weight: 600;
		font-size: 13px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
	}
	.results {
		padding: 16px 18px;
		border-top: 2px solid color-mix(in srgb, var(--primary) 55%, var(--border));
	}
	.col {
		padding: 14px 16px;
	}
	/* colored edges so the two retrieval arms read at a glance */
	.dense-col {
		border-top: 2px solid var(--dense);
		box-shadow: 0 0 26px -18px var(--dense);
	}
	.sparse-col {
		border-top: 2px solid var(--sparse);
		box-shadow: 0 0 26px -18px var(--sparse);
	}
	.clist {
		list-style: none;
		margin: 0;
		padding: 0;
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 12.5px;
	}
	.clist li {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 3px 0;
	}
	.clist .r {
		color: var(--text-faint);
		width: 18px;
	}
	.clist .d {
		color: var(--text-dim);
	}
	.clist .s {
		margin-left: auto;
	}

	/* ── empty / banners / footer ── */
	.empty {
		padding: 44px 32px;
		text-align: center;
	}
	.empty .big {
		font-size: 22px;
		font-weight: 600;
		margin-bottom: 8px;
	}
	.empty p {
		max-width: 620px;
		margin: 0 auto;
		font-size: 14px;
	}
	.banner {
		padding: 12px 16px;
		border-radius: var(--radius-sm);
		border: 1px solid var(--border);
	}
	.banner.err {
		border-color: var(--danger);
		color: #ffc0c0;
		background: rgba(255, 122, 122, 0.06);
	}
	.foot {
		display: flex;
		flex-direction: column;
		gap: 5px;
		font-size: 12.5px;
		margin-top: 8px;
		padding-top: 18px;
		border-top: 1px solid var(--border-soft);
	}
	.foot .lim {
		font-size: 12px;
		color: var(--text-faint);
	}

	@media (max-width: 900px) {
		.explain {
			grid-template-columns: 1fr;
		}
		.cand-row {
			grid-template-columns: 1fr;
		}
	}
</style>
