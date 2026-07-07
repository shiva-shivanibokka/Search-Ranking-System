<script lang="ts">
	import { search, type SearchResponse, type ResultItem } from '$lib/api';
	import { SAMPLE_QUERIES } from '$lib/samples';

	let query = $state('');
	let topK = $state(10);
	let loading = $state(false);
	let error = $state('');
	let lr = $state<SearchResponse | null>(null);
	let ce = $state<SearchResponse | null>(null);

	async function run(q?: string) {
		const text = (q ?? query).trim();
		if (!text || loading) return;
		query = text;
		loading = true;
		error = '';
		lr = null;
		ce = null;
		try {
			// Both rerankers run on the SAME fused candidates — only the rank stage differs.
			const [a, b] = await Promise.all([
				search({ query: text, top_k: topK, ranker: 'lambdarank' }),
				search({ query: text, top_k: topK, ranker: 'crossencoder' })
			]);
			lr = a;
			ce = b;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	// Where doc_id sits in the other ranker's list (1-indexed), or null if absent.
	function otherRank(other: SearchResponse | null, docId: number): number | null {
		const i = other?.results.findIndex((r) => r.doc_id === docId) ?? -1;
		return i >= 0 ? i + 1 : null;
	}

	// How much the two top-K lists agree (same doc at same rank) — a quick concordance read.
	const agreement = $derived.by(() => {
		if (!lr || !ce) return null;
		const n = Math.min(lr.results.length, ce.results.length);
		let same = 0;
		for (let i = 0; i < n; i++) if (lr.results[i].doc_id === ce.results[i].doc_id) same++;
		return n ? Math.round((same / n) * 100) : 0;
	});

	const columns = $derived(
		lr && ce
			? [
					{ r: lr, other: ce, name: 'LambdaRank', sub: 'GBDT · fast', cls: 'lr' },
					{ r: ce, other: lr, name: 'CrossEncoder', sub: 'neural · slower', cls: 'ce' }
				]
			: []
	);
</script>

<svelte:head><title>Compare rankers — Neural Search Ranking</title></svelte:head>

<main class="wrap page">
	<div class="head">
		<h1>Compare rankers</h1>
		<p class="dim">
			Same query, same fused candidates — <b>LambdaRank</b> (fast GBDT) vs <b>CrossEncoder</b> (slower
			neural) side by side. See how the orderings differ and the latency cost. For the <em>objective</em>
			"which wins" (measured NDCG@10 on the dev set), see
			<a
				href="https://github.com/shiva-shivanibokka/Search-Ranking-System#14-evaluation-results"
				target="_blank"
				rel="noreferrer">§14 of the README ↗</a
			>.
		</p>
	</div>

	<section class="search card">
		<div class="searchrow">
			<span class="mag" aria-hidden="true">⌕</span>
			<input
				class="q"
				placeholder="Ask something factual — e.g. how does a vaccine work"
				bind:value={query}
				onkeydown={(e) => e.key === 'Enter' && run()}
			/>
			<label class="kctl">Top&nbsp;K
				<select bind:value={topK}>{#each [5, 10, 20] as k}<option value={k}>{k}</option>{/each}</select>
			</label>
			<button class="btn" onclick={() => run()} disabled={loading || !query.trim()}>
				{loading ? 'Comparing…' : 'Compare'}
			</button>
		</div>
		<div class="chips">
			<span class="try dim">try</span>
			{#each SAMPLE_QUERIES as sq (sq)}
				<button class="chip" onclick={() => run(sq)}>{sq}</button>
			{/each}
		</div>
	</section>

	{#if error}
		<div class="banner err">⚠ {error}</div>
	{/if}

	{#if loading}
		<div class="empty card dim">Running both rerankers… (CrossEncoder is the slower of the two)</div>
	{:else if lr && ce}
		{#if agreement !== null}
			<div class="concord">
				<span class="pill mono">top-{Math.min(lr.results.length, ce.results.length)} agreement: {agreement}%</span>
				<span class="pill mono">LambdaRank {lr.timings.total_ms} ms</span>
				<span class="pill mono">CrossEncoder {ce.timings.total_ms} ms</span>
			</div>
		{/if}
		<div class="cols">
			{#each columns as col (col.cls)}
				<section class="card col {col.cls}">
					<div class="colhead">
						<div>
							<span class="rname">{col.name}</span>
							<span class="rsub dim">{col.sub}</span>
						</div>
						<span class="lat mono">{col.r.timings.total_ms} ms · rerank {col.r.timings.rerank_ms} ms</span>
					</div>
					{#each col.r.results as item, i (item.doc_id)}
						{@const o = otherRank(col.other, item.doc_id)}
						<div class="row">
							<span class="rk mono">{i + 1}</span>
							<div class="body">
								<div class="meta">
									<span class="doc mono">doc {item.doc_id}</span>
									<span class="sc mono">{item.score.toFixed(3)}</span>
									{#if o === null}
										<span class="delta only">only here</span>
									{:else if o === i + 1}
										<span class="delta same">= same rank</span>
									{:else if o > i + 1}
										<span class="delta up" title="ranked {o - (i + 1)} higher here than by the other ranker">▲ {o - (i + 1)}</span>
									{:else}
										<span class="delta down" title="ranked {i + 1 - o} lower here than by the other ranker">▼ {i + 1 - o}</span>
									{/if}
								</div>
								<p class="text">{item.text}</p>
							</div>
						</div>
					{/each}
				</section>
			{/each}
		</div>
		<p class="dim foot">
			<b>▲</b> this doc is ranked higher here than by the other ranker · <b>▼</b> lower ·
			<b>only here</b> not in the other's top&nbsp;{topK}. High agreement ⇒ the two models mostly
			concur; big shifts ⇒ the CrossEncoder is re-judging relevance differently from the GBDT.
		</p>
	{:else}
		<div class="empty card">
			<div class="big">LambdaRank vs CrossEncoder, head-to-head</div>
			<p class="dim">Enter a query or pick one above. Both rerankers score the same fused candidates, so any difference is purely the ranking model — shown with per-doc rank shifts and each side's latency.</p>
		</div>
	{/if}
</main>

<style>
	.page {
		display: flex;
		flex-direction: column;
		gap: 16px;
		padding-top: 22px;
		padding-bottom: 40px;
	}
	.head h1 {
		margin: 0 0 6px;
		font-size: 24px;
	}
	.head p {
		margin: 0;
		max-width: 820px;
		font-size: 14px;
	}
	.search {
		padding: 16px 18px;
		display: flex;
		flex-direction: column;
		gap: 14px;
		border-top: 2px solid color-mix(in srgb, var(--primary) 55%, var(--border));
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
		padding: 13px 15px;
		font-size: 16px;
	}
	.q:focus {
		border-color: var(--primary);
		outline: none;
	}
	.kctl {
		display: inline-flex;
		gap: 6px;
		align-items: center;
		font-size: 13px;
		color: var(--text-dim);
	}
	.kctl select {
		background: var(--surface-2);
		border: 1px solid var(--border);
		color: var(--text);
		border-radius: var(--radius-xs);
		padding: 7px 9px;
	}
	.chips {
		display: flex;
		gap: 7px;
		flex-wrap: wrap;
		align-items: center;
	}
	.try {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
	}
	.chip {
		background: var(--surface-2);
		border: 1px solid var(--border);
		color: var(--text-dim);
		border-radius: 999px;
		padding: 5px 12px;
		font-size: 12.5px;
	}
	.chip:hover {
		color: var(--text);
		border-color: var(--primary);
	}
	.concord {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}
	.cols {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
		align-items: start;
	}
	.col {
		padding: 14px 16px;
	}
	.col.lr {
		border-top: 2px solid var(--primary-2);
	}
	.col.ce {
		border-top: 2px solid var(--good);
	}
	.colhead {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: 10px;
		margin-bottom: 10px;
		padding-bottom: 8px;
		border-bottom: 1px solid var(--border-soft);
	}
	.rname {
		font-weight: 700;
		font-size: 15px;
	}
	.rsub {
		font-size: 12px;
		margin-left: 6px;
	}
	.lat {
		font-size: 11.5px;
		color: var(--text-faint);
	}
	.row {
		display: flex;
		gap: 12px;
		padding: 10px 0;
		border-bottom: 1px solid var(--border-soft);
	}
	.row:last-child {
		border-bottom: none;
	}
	.rk {
		flex: 0 0 26px;
		height: 26px;
		border-radius: 6px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 12px;
		color: var(--text-dim);
	}
	.body {
		flex: 1;
		min-width: 0;
	}
	.meta {
		display: flex;
		gap: 8px;
		align-items: center;
		flex-wrap: wrap;
		margin-bottom: 4px;
		font-size: 12px;
	}
	.doc {
		color: var(--text-faint);
	}
	.sc {
		color: var(--text-dim);
	}
	.delta {
		font-family: var(--mono);
		font-size: 11px;
		padding: 1px 6px;
		border-radius: 999px;
		border: 1px solid var(--border);
	}
	.delta.up {
		color: var(--good);
		border-color: color-mix(in srgb, var(--good) 45%, var(--border));
	}
	.delta.down {
		color: var(--danger);
		border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
	}
	.delta.only {
		color: var(--sparse);
		border-color: color-mix(in srgb, var(--sparse) 45%, var(--border));
	}
	.delta.same {
		color: var(--text-faint);
	}
	.text {
		margin: 0;
		font-size: 13.5px;
		color: color-mix(in srgb, var(--text) 92%, transparent);
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
	.empty {
		padding: 40px 28px;
		text-align: center;
	}
	.empty .big {
		font-size: 20px;
		font-weight: 600;
		margin-bottom: 8px;
	}
	.empty p {
		max-width: 600px;
		margin: 0 auto;
		font-size: 14px;
	}
	.foot {
		font-size: 12.5px;
		padding-top: 6px;
	}
	@media (max-width: 820px) {
		.cols {
			grid-template-columns: 1fr;
		}
	}
</style>
