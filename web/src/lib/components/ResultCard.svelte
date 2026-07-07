<script lang="ts">
	import type { ResultItem } from '$lib/api';
	let { item, citation }: { item: ResultItem; citation?: number } = $props();
</script>

<div class="result">
	<div class="rank mono">{item.rank}</div>
	<div class="body">
		<div class="meta">
			{#if citation}<span class="cite mono">[{citation}]</span>{/if}
			<span class="doc mono">doc {item.doc_id}</span>
			<span class="score mono">{item.score.toFixed(3)}</span>
		</div>
		<p class="text">{item.text}</p>
	</div>
</div>

<style>
	.result {
		display: flex;
		gap: 14px;
		padding: 14px 0;
		border-bottom: 1px solid var(--border-soft);
	}
	.result:last-child {
		border-bottom: none;
	}
	.rank {
		flex: 0 0 30px;
		height: 30px;
		border-radius: var(--radius-xs);
		background: var(--surface-2);
		border: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 13px;
		color: var(--text-dim);
		font-weight: 500;
	}
	.body {
		flex: 1;
		min-width: 0;
	}
	.meta {
		display: flex;
		gap: 10px;
		align-items: center;
		margin-bottom: 5px;
		font-size: 12px;
	}
	.cite {
		color: var(--primary-2);
		font-weight: 700;
	}
	.doc {
		color: var(--text-faint);
	}
	.score {
		color: var(--dense);
		margin-left: auto;
	}
	.text {
		margin: 0;
		font-size: 14px;
		color: color-mix(in srgb, var(--text) 92%, transparent);
	}
</style>
