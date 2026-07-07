<script lang="ts">
	import '../app.css';
	import { page } from '$app/stores';
	let { children } = $props();
	const path = $derived($page.url.pathname);
</script>

<nav class="topnav">
	<div class="wrap navrow">
		<a class="brand" href="/">
			<span class="glyph" aria-hidden="true"></span>
			<span class="word">Neural&nbsp;Search<span class="accent">·</span>Ranking</span>
		</a>
		<div class="tabs">
			<a class="tab" class:active={path === '/'} href="/">Search</a>
			<a class="tab" class:active={path === '/compare'} href="/compare">Compare</a>
		</div>
		<a
			class="gh"
			href="https://github.com/shiva-shivanibokka/Search-Ranking-System"
			target="_blank"
			rel="noreferrer">GitHub ↗</a
		>
	</div>
</nav>

{@render children()}

<style>
	.topnav {
		border-bottom: 1px solid var(--border-soft);
		background: rgba(10, 14, 28, 0.72);
		backdrop-filter: blur(8px);
		position: sticky;
		top: 0;
		z-index: 20;
	}
	.navrow {
		display: flex;
		align-items: center;
		gap: 18px;
		padding-top: 12px;
		padding-bottom: 12px;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
		font-weight: 700;
		font-size: 17px;
		letter-spacing: -0.02em;
		color: var(--text);
		text-decoration: none;
	}
	.glyph {
		width: 20px;
		height: 20px;
		border-radius: 6px;
		background: conic-gradient(from 140deg, var(--dense), var(--primary), var(--sparse), var(--dense));
		box-shadow: 0 0 18px -4px var(--primary);
		flex: 0 0 auto;
	}
	.word .accent {
		color: var(--primary-2);
		margin: 0 2px;
	}
	.tabs {
		display: flex;
		gap: 4px;
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: 999px;
		padding: 3px;
	}
	.tab {
		padding: 6px 16px;
		border-radius: 999px;
		font-size: 13px;
		font-weight: 600;
		color: var(--text-dim);
		text-decoration: none;
		transition: all 0.14s ease;
	}
	.tab:hover {
		color: var(--text);
	}
	.tab.active {
		background: var(--primary);
		color: #fff;
		box-shadow: 0 4px 14px -6px var(--primary);
	}
	.gh {
		margin-left: auto;
		font-size: 13px;
		color: var(--text-dim);
		text-decoration: none;
	}
	.gh:hover {
		color: var(--text);
	}
	@media (max-width: 560px) {
		.word {
			display: none;
		}
	}
</style>
