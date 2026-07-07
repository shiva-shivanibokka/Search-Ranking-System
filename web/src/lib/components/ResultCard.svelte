<script lang="ts">
	import type { ResultItem } from '$lib/api';
	let { item, citation }: { item: ResultItem; citation?: number } = $props();
</script>

<div class="result">
	<div class="rank">{item.rank}</div>
	<div class="body">
		<div class="meta">
			{#if citation}<span class="cite">[{citation}]</span>{/if}
			<span class="pill mono">doc {item.doc_id}</span>
			<span class="pill mono">{item.ranker} · {item.score.toFixed(3)}</span>
		</div>
		<p class="text">{item.text}</p>
	</div>
</div>

<style>
	.result {
		display: flex;
		gap: 12px;
		padding: 12px 0;
		border-bottom: 1px solid var(--border);
	}
	.rank {
		flex: 0 0 28px;
		height: 28px;
		border-radius: 6px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: center;
		font-family: var(--mono);
		font-size: 13px;
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
		margin-bottom: 5px;
	}
	.cite {
		font-family: var(--mono);
		font-size: 12px;
		color: var(--accent-2);
		font-weight: 700;
	}
	.text {
		margin: 0;
		font-size: 14px;
	}
</style>
