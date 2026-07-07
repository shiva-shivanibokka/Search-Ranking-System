<script lang="ts">
	import type { SearchResponse } from '$lib/api';
	let { data }: { data: SearchResponse } = $props();

	const s = $derived(data.stages);
	const t = $derived(data.timings);
</script>

<div class="card breakdown">
	<div class="head">
		<strong>How this was ranked</strong>
		<span class="pill mono">{t.total_ms} ms total</span>
	</div>

	<div class="pipe">
		<div class="stage">
			<div class="stage-title">1 · Understand</div>
			<div class="stage-body">
				intent <code>{s.intent}</code>
				<br />
				HyDE {s.hyde_used ? 'on' : 'off'}
				<span class="ms">{t.hyde_ms} ms</span>
			</div>
		</div>
		<div class="arrow">→</div>
		<div class="stage">
			<div class="stage-title">2 · Retrieve (hybrid)</div>
			<div class="stage-body">
				FAISS dense + BM25 sparse
				<br />
				fused via RRF → {s.fused_count} candidates
				<span class="ms">{t.retrieve_ms} ms</span>
			</div>
		</div>
		<div class="arrow">→</div>
		<div class="stage">
			<div class="stage-title">3 · Rerank</div>
			<div class="stage-body">
				<code>{data.ranker}</code>
				<br />
				top {data.results.length} shown
				<span class="ms">{t.rerank_ms} ms</span>
			</div>
		</div>
	</div>

	{#if s.hyde_used}
		<div class="hyde">
			<span class="dim">HyDE expansion embedded:</span>
			<em>"{s.embed_text_preview}…"</em>
		</div>
	{/if}

	<div class="cols">
		<div>
			<div class="col-title">Top dense (FAISS) candidates</div>
			<ul>
				{#each s.dense_top as c (c.doc_id)}
					<li><span class="mono">#{c.rank}</span> doc {c.doc_id} <span class="dim">· {c.score.toFixed(3)}</span></li>
				{/each}
			</ul>
		</div>
		<div>
			<div class="col-title">Top sparse (BM25) candidates</div>
			<ul>
				{#each s.sparse_top as c (c.doc_id)}
					<li><span class="mono">#{c.rank}</span> doc {c.doc_id} <span class="dim">· {c.score.toFixed(2)}</span></li>
				{/each}
			</ul>
		</div>
	</div>
</div>

<style>
	.breakdown {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.head {
		display: flex;
		justify-content: space-between;
		align-items: center;
	}
	.pipe {
		display: flex;
		align-items: stretch;
		gap: 8px;
		flex-wrap: wrap;
	}
	.stage {
		flex: 1 1 160px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 10px 12px;
	}
	.stage-title {
		font-weight: 700;
		font-size: 13px;
		margin-bottom: 4px;
	}
	.stage-body {
		font-size: 13px;
		color: var(--text-dim);
	}
	.ms {
		display: inline-block;
		margin-top: 4px;
		font-family: var(--mono);
		font-size: 11px;
		color: var(--accent);
	}
	.arrow {
		display: flex;
		align-items: center;
		color: var(--text-dim);
		font-size: 20px;
	}
	.hyde {
		font-size: 13px;
		border-left: 2px solid var(--accent-2);
		padding-left: 10px;
	}
	.cols {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
	}
	.col-title {
		font-size: 12px;
		font-weight: 700;
		color: var(--text-dim);
		margin-bottom: 6px;
	}
	ul {
		margin: 0;
		padding: 0;
		list-style: none;
		font-size: 13px;
		display: flex;
		flex-direction: column;
		gap: 3px;
	}
	@media (max-width: 620px) {
		.cols {
			grid-template-columns: 1fr;
		}
		.arrow {
			display: none;
		}
	}
</style>
