# 0003 — Provider-agnostic LLM layer with a zero-key default

**Status:** Accepted

## Context
Query understanding (intent, rewriting, HyDE) uses an LLM. The original code
hardcoded Anthropic Claude, which (a) locks us to one vendor, (b) requires a paid
key for the system to function at all, and (c) makes a public free demo a cost/
abuse risk.

## Decision
Introduce `services/shared/llm.py`: one `LLMProvider.complete()` interface with
swappable backends — **Groq, Gemini, OpenAI, Anthropic** — selected by the
`LLM_PROVIDER` env var. SDKs import lazily; a missing key/SDK degrades to a
**zero-key `NoneProvider`** instead of crashing. Default is `none`, so the system
runs at $0 with rule-based query understanding only.

## Consequences
- **Pro:** no vendor lock-in; switch providers with one env var.
- **Pro:** the free demo works with zero keys; Groq/Gemini free tiers give real
  LLM behavior for free; Claude/OpenAI available when a paid key is set.
- **Pro:** graceful degradation — the service always starts.
- **Trade-off:** the lowest common denominator interface (system + user → text)
  doesn't expose provider-specific features (tool use, structured output). Fine
  for this use case; extend the interface if richer calls are needed.

## At 10× scale
Add response caching keyed on (provider, prompt) and a token/budget guard;
consider routing cheap calls (intent) to a small/free model and only HyDE to a
stronger one.
