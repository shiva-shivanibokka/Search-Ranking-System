<script lang="ts">
	import { PROVIDERS, PROVIDER_ORDER, type ProviderId } from '$lib/providers';
	import { byok } from '$lib/stores';
	import { generateAnswer } from '$lib/rag';
	import type { ResultItem } from '$lib/api';

	let { query, passages }: { query: string; passages: ResultItem[] } = $props();

	let showKey = $state(false);
	let answer = $state('');
	let error = $state('');
	let loading = $state(false);

	const provider = $derived($byok.provider);
	const info = $derived(PROVIDERS[provider]);
	const selectedModel = $derived($byok.models[provider] ?? info.models[0].id);
	const isCustom = $derived(!info.models.some((m) => m.id === selectedModel));
	const key = $derived(($byok.keys[provider] ?? '').trim());
	const ready = $derived(key.length > 0 && selectedModel.length > 0 && passages.length > 0);
	const MAX_PASSAGES = 5;

	function setProvider(p: ProviderId) {
		byok.update((s) => ({ ...s, provider: p }));
	}
	function setModel(id: string) {
		byok.update((s) => ({ ...s, models: { ...s.models, [provider]: id } }));
	}
	function setKey(v: string) {
		byok.update((s) => ({ ...s, keys: { ...s.keys, [provider]: v } }));
	}

	async function run() {
		if (!ready || loading) return;
		loading = true;
		error = '';
		answer = '';
		try {
			answer = await generateAnswer({
				provider,
				model: selectedModel,
				apiKey: key,
				query,
				passages,
				maxPassages: MAX_PASSAGES
			});
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	const rendered = $derived(
		answer.split(/(\[\d+\])/g).map((p) => ({ p, cite: /^\[\d+\]$/.test(p) }))
	);
</script>

<section class="rag card">
	<div class="controls">
		<div class="field">
			<span class="lab">Provider</span>
			<select value={provider} onchange={(e) => setProvider((e.target as HTMLSelectElement).value as ProviderId)}>
				{#each PROVIDER_ORDER as p (p)}
					<option value={p}>{PROVIDERS[p].label}{($byok.keys[p] ?? '').trim() ? '  •' : ''}</option>
				{/each}
			</select>
		</div>

		<div class="field">
			<span class="lab">Model</span>
			<select
				value={isCustom ? '__custom__' : selectedModel}
				onchange={(e) => {
					const v = (e.target as HTMLSelectElement).value;
					setModel(v === '__custom__' ? '' : v);
				}}
			>
				{#each info.models as m (m.id)}
					<option value={m.id}>{m.label}{m.free ? '  (free)' : ''}</option>
				{/each}
				<option value="__custom__">Custom model id…</option>
			</select>
		</div>

		{#if isCustom}
			<div class="field grow">
				<span class="lab">Custom model</span>
				<input
					type="text"
					placeholder="exact model id"
					value={selectedModel}
					oninput={(e) => setModel((e.target as HTMLInputElement).value)}
				/>
			</div>
		{/if}

		<div class="field grow">
			<span class="lab">{info.label} API key</span>
			<div class="keyrow">
				<input
					type={showKey ? 'text' : 'password'}
					placeholder="paste your key — it stays in your browser"
					value={key}
					autocomplete="off"
					spellcheck="false"
					oninput={(e) => setKey((e.target as HTMLInputElement).value)}
				/>
				<button class="mini" type="button" onclick={() => (showKey = !showKey)}>
					{showKey ? 'hide' : 'show'}
				</button>
			</div>
		</div>

		<button class="btn gen" onclick={run} disabled={!ready || loading}>
			{loading ? 'Generating…' : 'Generate answer'}
		</button>
	</div>

	<div class="meta">
		<span class="lock">🔒 key never leaves your browser</span>
		<a href={info.keyUrl} target="_blank" rel="noreferrer">Get a {info.label} key ↗</a>
		<span class="dim">·  {info.keyHint}</span>
	</div>

	{#if error}
		<div class="answer err">⚠ {error}</div>
	{:else if answer}
		<div class="answer">
			{#each rendered as r}{#if r.cite}<span class="cite">{r.p}</span>{:else}{r.p}{/if}{/each}
			<div class="src">answered by {info.label} · {selectedModel} · grounded in the top {Math.min(MAX_PASSAGES, passages.length)} results</div>
		</div>
	{:else if passages.length === 0}
		<div class="answer idle dim">Run a search, then generate a cited answer from the results — using your own model.</div>
	{:else if !key}
		<div class="answer idle dim">Add a {info.label} key above to generate a grounded RAG answer over these results.</div>
	{/if}
</section>

<style>
	.rag {
		padding: 16px 18px;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 12px;
		align-items: flex-end;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 5px;
		min-width: 150px;
	}
	.field.grow {
		flex: 1 1 220px;
	}
	.lab {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.08em;
		color: var(--text-faint);
	}
	select,
	input {
		background: var(--bg);
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 10px 12px;
		width: 100%;
	}
	select {
		cursor: pointer;
	}
	.keyrow {
		display: flex;
		gap: 6px;
	}
	.mini {
		background: var(--surface-2);
		border: 1px solid var(--border);
		border-radius: var(--radius-xs);
		padding: 0 12px;
		font-size: 12px;
		color: var(--text-dim);
		white-space: nowrap;
	}
	.gen {
		flex: 0 0 auto;
		align-self: flex-end;
		height: 42px;
	}
	.meta {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
		align-items: center;
		font-size: 12px;
		color: var(--text-dim);
	}
	.lock {
		color: var(--good);
		font-weight: 500;
	}
	.answer {
		background: var(--bg);
		border: 1px solid var(--border-soft);
		border-left: 3px solid var(--primary);
		border-radius: var(--radius-sm);
		padding: 14px 16px;
		line-height: 1.65;
		white-space: pre-wrap;
	}
	.answer.idle {
		border-left-color: var(--border);
		font-size: 13.5px;
	}
	.answer.err {
		border-left-color: var(--danger);
		color: #ffc0c0;
	}
	.cite {
		font-family: var(--mono);
		font-size: 12px;
		font-weight: 700;
		color: var(--primary-2);
		padding: 0 1px;
	}
	.src {
		margin-top: 10px;
		font-size: 11.5px;
		color: var(--text-faint);
		font-family: var(--mono);
	}
</style>
