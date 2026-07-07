<script lang="ts">
	import type { SearchResponse } from '$lib/api';
	let { data }: { data: SearchResponse } = $props();
	const s = $derived(data.stages);
	const t = $derived(data.timings);
</script>

<section class="rail card">
	<div class="head">
		<span class="title">Pipeline</span>
		<span class="total mono">{t.total_ms} ms</span>
	</div>

	<div class="track">
		<div class="node understand">
			<div class="idx mono">01</div>
			<div class="name">Understand</div>
			<div class="detail">
				intent <code>{s.intent}</code> · HyDE {s.hyde_used ? 'on' : 'off'}
			</div>
			<div class="t mono">{t.hyde_ms} ms</div>
		</div>

		<div class="link"></div>

		<div class="node retrieve">
			<div class="idx mono">02</div>
			<div class="name">Retrieve <span class="hy">hybrid</span></div>
			<div class="detail">
				<span class="dense">FAISS</span> + <span class="sparse">BM25</span> → RRF · {s.fused_count} candidates
			</div>
			<div class="t mono">{t.retrieve_ms} ms</div>
		</div>

		<div class="link"></div>

		<div class="node rank">
			<div class="idx mono">03</div>
			<div class="name">Rank</div>
			<div class="detail"><code>{data.ranker}</code> · top {data.results.length}</div>
			<div class="t mono">{t.rerank_ms} ms</div>
		</div>

		<div class="link"></div>

		<div class="node answer">
			<div class="idx mono">04</div>
			<div class="name">Answer</div>
			<div class="detail">client-side BYOK RAG</div>
			<div class="t mono">in browser</div>
		</div>
	</div>

	{#if s.hyde_used}
		<div class="hyde">
			<span class="dim">HyDE embedded a hypothetical answer:</span>
			<em>"{s.embed_text_preview}…"</em>
		</div>
	{/if}
</section>

<style>
	.rail {
		padding: 16px 18px;
	}
	.head {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		margin-bottom: 12px;
	}
	.title {
		font-weight: 600;
		font-size: 13px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		color: var(--text-dim);
	}
	.total {
		font-size: 13px;
		color: var(--primary-2);
		font-weight: 700;
	}
	.track {
		display: flex;
		align-items: stretch;
		gap: 0;
	}
	.node {
		flex: 1 1 0;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 12px 14px;
		position: relative;
		min-width: 0;
	}
	.idx {
		font-size: 11px;
		color: var(--text-faint);
		font-weight: 700;
	}
	.name {
		font-weight: 600;
		font-size: 15px;
		margin: 2px 0 4px;
	}
	.hy {
		font-size: 10px;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		color: var(--bg);
		background: linear-gradient(90deg, var(--dense), var(--sparse));
		padding: 1px 6px;
		border-radius: 999px;
		vertical-align: middle;
		font-weight: 700;
	}
	.detail {
		font-size: 12.5px;
		color: var(--text-dim);
		line-height: 1.4;
	}
	.dense {
		color: var(--dense);
		font-weight: 600;
	}
	.sparse {
		color: var(--sparse);
		font-weight: 600;
	}
	.t {
		margin-top: 8px;
		font-size: 12px;
		color: var(--text-faint);
	}
	/* stage accents */
	.retrieve {
		border-image: linear-gradient(90deg, var(--dense), var(--sparse)) 1;
	}
	.rank {
		border-color: color-mix(in srgb, var(--primary) 55%, var(--border));
	}
	.answer {
		border-color: color-mix(in srgb, var(--good) 45%, var(--border));
	}
	.link {
		flex: 0 0 26px;
		align-self: center;
		height: 2px;
		background: linear-gradient(90deg, var(--border), var(--primary), var(--border));
		opacity: 0.8;
	}
	.hyde {
		margin-top: 12px;
		font-size: 12.5px;
		border-left: 2px solid var(--primary);
		padding-left: 10px;
		color: var(--text);
	}
	@media (max-width: 760px) {
		.track {
			flex-direction: column;
			gap: 8px;
		}
		.link {
			width: 2px;
			height: 16px;
			align-self: center;
		}
	}
</style>
