"""
Provider-agnostic LLM layer.

Query Understanding (intent classification, query rewriting, HyDE) needs an LLM,
but should not be locked to a single vendor. This module exposes one interface,
``LLMProvider.complete()``, with swappable backends selected at runtime by the
``LLM_PROVIDER`` environment variable:

    LLM_PROVIDER = groq | gemini | openai | anthropic | none   (default: none)

Design choices:
  * Each backend lazily imports its SDK, so a missing SDK only fails the provider
    that needs it — not the whole service.
  * ``none`` is the zero-key default. It makes every LLM call a no-op, so the
    public demo runs at $0 with no API keys. Query Understanding degrades to
    rule-based intent only (no rewrite, no HyDE).
  * Groq and Gemini both have genuinely free API tiers — the recommended choice
    for a free-but-real LLM-powered demo. OpenAI and Anthropic are paid.

Every backend implements ``complete(system, user, max_tokens, temperature) -> str``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMUnavailable(RuntimeError):
    """Raised when a selected provider cannot be used (missing key/SDK)."""


class LLMProvider(ABC):
    """One text-in/text-out completion call, vendor-agnostic."""

    name: str = "base"

    @property
    def available(self) -> bool:
        """Whether this provider can actually serve calls (key + SDK present)."""
        return True

    @abstractmethod
    def complete(
        self, system: str, user: str, *, max_tokens: int = 256, temperature: float = 0.0
    ) -> str:
        ...


class NoneProvider(LLMProvider):
    """Zero-key fallback. Signals 'no LLM' so callers skip LLM-only steps."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system, user, *, max_tokens=256, temperature=0.0) -> str:
        raise LLMUnavailable(
            "No LLM provider configured (LLM_PROVIDER=none). "
            "Set LLM_PROVIDER to groq|gemini|openai|anthropic and provide its API key."
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        import anthropic

        self._key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("LLM_MODEL", "claude-haiku-4-5")
        self._client = anthropic.Anthropic(api_key=self._key) if self._key else None

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system, user, *, max_tokens=256, temperature=0.0) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        from openai import OpenAI

        self._key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=self._key) if self._key else None

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system, user, *, max_tokens=256, temperature=0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


class GroqProvider(LLMProvider):
    """Groq — free API tier, OpenAI-compatible, very fast. Good free default."""

    name = "groq"

    def __init__(self) -> None:
        from groq import Groq

        self._key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        self._client = Groq(api_key=self._key) if self._key else None

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(self, system, user, *, max_tokens=256, temperature=0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()


class GeminiProvider(LLMProvider):
    """Google Gemini — free API tier."""

    name = "gemini"

    def __init__(self) -> None:
        import google.generativeai as genai

        self._key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        if self._key:
            genai.configure(api_key=self._key)
            self._genai = genai
        else:
            self._genai = None

    @property
    def available(self) -> bool:
        return self._genai is not None

    def complete(self, system, user, *, max_tokens=256, temperature=0.0) -> str:
        model = self._genai.GenerativeModel(
            self.model_name, system_instruction=system
        )
        resp = model.generate_content(
            user,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return resp.text.strip()


_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "none": NoneProvider,
}


def get_llm_provider() -> LLMProvider:
    """Instantiate the provider named by LLM_PROVIDER.

    Falls back to NoneProvider when the provider is unknown, its SDK is missing,
    or its API key is absent — so the service always starts and degrades to
    rule-based behavior instead of crashing.
    """
    name = os.getenv("LLM_PROVIDER", "none").lower().strip()
    cls = _PROVIDERS.get(name)
    if cls is None:
        return NoneProvider()
    try:
        provider = cls()
    except Exception:
        # SDK import failed or constructor errored — degrade to no-LLM.
        return NoneProvider()
    if not provider.available:
        return NoneProvider()
    return provider
