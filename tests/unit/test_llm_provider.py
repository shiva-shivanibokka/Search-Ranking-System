"""Unit tests for the provider-agnostic LLM layer."""

import pytest

from services.shared.llm import (
    LLMUnavailable,
    NoneProvider,
    get_llm_provider,
)


def test_default_provider_is_none(monkeypatch):
    """With no LLM_PROVIDER set, we degrade to the zero-key NoneProvider."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = get_llm_provider()
    assert isinstance(provider, NoneProvider)
    assert provider.available is False


def test_unknown_provider_degrades_to_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-real-provider")
    provider = get_llm_provider()
    assert isinstance(provider, NoneProvider)


def test_none_provider_raises_on_complete():
    provider = NoneProvider()
    with pytest.raises(LLMUnavailable):
        provider.complete("system", "user")


def test_paid_provider_without_key_degrades_to_none(monkeypatch):
    """Selecting a provider without supplying its key must not crash — it degrades."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = get_llm_provider()
    # Either anthropic SDK missing or no key → NoneProvider, never an exception.
    assert provider.available is False
