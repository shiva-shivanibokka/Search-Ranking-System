<script lang="ts">
	import type { SearchResponse } from '$lib/api';
	let { data = null, loading = false }: { data?: SearchResponse | null; loading?: boolean } = $props();
	const s = $derived(data?.stages ?? null);
	const t = $derived(data?.timings ?? null);
</script>

<section class="rail card" class:running={loading}>
	<div class="head">
		<span class="title">Pipeline{#if loading}<span class="run"> · running…</span>{/if}</span>
		{#if t}<span class="total mono">{t.total_ms} ms</span>{/if}
	</div>

	<div class="track">
		<div class="node understand" style="--i:0">
			<div class="top">
				<span class="idx mono">01</span>
				<span class="info" tabindex="0" role="button" aria-label="What the Understand stage does">?
					<span class="tip" role="tooltip">Figures out what you're asking (its intent) and can rewrite it into a fuller form — <b>HyDE</b> — so the search matches meaning, not just your exact words.</span>
				</span>
			</div>
			<div class="name">Understand</div>
			<div class="detail">
				{#if s}intent <code>{s.intent}</code> · HyDE {s.hyde_used ? 'on' : 'off'}{:else}reading the query…{/if}
			</div>
			<div class="t mono">{t ? `${t.hyde_ms} ms` : ''}</div>
		</div>

		<div class="link"></div>

		<div class="node retrieve" style="--i:1">
			<div class="top">
				<span class="idx mono">02</span>
				<span class="info" tabindex="0" role="button" aria-label="What the Retrieve stage does">?
					<span class="tip" role="tooltip">Two searches run at once — a neural one that matches <b>meaning</b> (<span class="dense">dense/FAISS</span>) and a <b>keyword</b> one (<span class="sparse">sparse/BM25</span>). Their two lists are merged into one candidate set (RRF).</span>
				</span>
			</div>
			<div class="name">Retrieve <span class="hy">hybrid</span></div>
			<div class="detail">
				{#if s}<span class="dense">FAISS</span> + <span class="sparse">BM25</span> → RRF · {s.fused_count} candidates{:else}searching dense + sparse…{/if}
			</div>
			<div class="t mono">{t ? `${t.retrieve_ms} ms` : ''}</div>
		</div>

		<div class="link"></div>

		<div class="node rank" style="--i:2">
			<div class="top">
				<span class="idx mono">03</span>
				<span class="info" tabindex="0" role="button" aria-label="What the Rank stage does">?
					<span class="tip" role="tooltip">A trained model (LambdaRank) re-orders the candidates by how relevant they really are, using signals like keyword overlap, match scores and passage length.</span>
				</span>
			</div>
			<div class="name">Rank</div>
			<div class="detail">{#if data}<code>{data.ranker}</code> · top {data.results.length}{:else}reranking candidates…{/if}</div>
			<div class="t mono">{t ? `${t.rerank_ms} ms` : ''}</div>
		</div>

		<div class="link"></div>

		<div class="node answer" style="--i:3">
			<div class="top">
				<span class="idx mono">04</span>
				<span class="info" tabindex="0" role="button" aria-label="What the Answer stage does">?
					<span class="tip" role="tooltip">Optional: your chosen AI model reads the top results and writes a short, cited answer — running <b>in your browser</b> with <b>your</b> key, which never reaches the server.</span>
				</span>
			</div>
			<div class="name">Answer</div>
			<div class="detail">client-side BYOK RAG</div>
			<div class="t mono">in browser</div>
		</div>
	</div>

	{#if s?.hyde_used}
		<div class="hyde">
			<span class="dim">HyDE embedded a hypothetical answer:</span>
			<em>"{s.embed_text_preview}…"</em>
		</div>
	{/if}
</section>

<style>
	.rail {
		padding: 16px 18px;
		overflow: visible;
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
	.run {
		color: var(--primary-2);
		letter-spacing: 0;
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
		padding: 10px 14px 12px;
		position: relative;
		min-width: 0;
		transition: box-shadow 0.25s ease;
	}
	.top {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 2px;
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
		min-height: 15px;
	}

	/* static stage identities */
	.retrieve {
		border-image: linear-gradient(90deg, var(--dense), var(--sparse)) 1;
	}
	.rank {
		border-color: color-mix(in srgb, var(--primary) 50%, var(--border));
	}
	.answer {
		border-color: color-mix(in srgb, var(--good) 42%, var(--border));
	}

	.link {
		flex: 0 0 26px;
		align-self: center;
		height: 2px;
		background: linear-gradient(90deg, var(--border), var(--primary), var(--border));
		opacity: 0.7;
	}

	/* ── flow animation while a query runs ── */
	.running .node {
		animation: flow 1.8s ease-in-out infinite;
		animation-delay: calc(var(--i) * 0.28s);
	}
	.running .link {
		background-size: 200% 100%;
		animation: slide 1.8s linear infinite;
	}
	@keyframes flow {
		0%,
		70%,
		100% {
			box-shadow: none;
		}
		30% {
			box-shadow:
				0 0 0 1px var(--primary),
				0 0 22px -4px var(--primary);
		}
	}
	@keyframes slide {
		to {
			background-position: -200% 0;
		}
	}

	/* ── info tooltip ── */
	.info {
		width: 18px;
		height: 18px;
		border-radius: 50%;
		border: 1px solid var(--border);
		background: var(--surface-3);
		color: var(--text-dim);
		font-size: 11px;
		font-weight: 700;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		cursor: help;
		position: relative;
		flex: 0 0 auto;
	}
	.info:hover,
	.info:focus-visible {
		color: var(--text);
		border-color: var(--primary);
	}
	.tip {
		position: absolute;
		top: calc(100% + 8px);
		right: -6px;
		width: 240px;
		background: #0c1124;
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 10px 12px;
		font-size: 12px;
		font-weight: 400;
		line-height: 1.45;
		color: var(--text);
		box-shadow: 0 16px 40px -12px rgba(0, 0, 0, 0.8);
		opacity: 0;
		visibility: hidden;
		transform: translateY(-4px);
		transition:
			opacity 0.14s ease,
			transform 0.14s ease;
		z-index: 30;
		text-transform: none;
		letter-spacing: 0;
	}
	.info:hover .tip,
	.info:focus-within .tip,
	.info:focus .tip {
		opacity: 1;
		visibility: visible;
		transform: translateY(0);
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
		.tip {
			right: auto;
			left: 0;
		}
	}
</style>
