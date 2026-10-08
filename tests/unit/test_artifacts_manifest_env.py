"""An empty HF_ARTIFACTS_REPO must fall back, not override the default with "".

`.github/workflows/retrain.yml` sets

    HF_ARTIFACTS_REPO: ${{ secrets.HF_ARTIFACTS_REPO }}

and an unset repository secret interpolates to the empty string rather than
being omitted. `os.getenv(name, default)` only returns the default when the
variable is *unset*, so the empty string won every scheduled run: after
installing CPU torch and the full requirements, `scripts/bootstrap.py` reached
the Hub and died with

    HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.',
    '--' and '..' are forbidden ... max length is 96: ''

These tests pin the distinction between unset, empty, and genuinely set.
"""
from __future__ import annotations

import importlib

import pytest

DEFAULT_REPO = "shiva-1993/search-ranking-system"
DEFAULT_REVISION = "main"


def _reload(monkeypatch, **env):
    """Re-import the manifest with a specific environment."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    import scripts.artifacts_manifest as manifest

    return importlib.reload(manifest)


def test_unset_falls_back_to_the_default(monkeypatch):
    m = _reload(monkeypatch, HF_ARTIFACTS_REPO=None, HF_ARTIFACTS_REVISION=None)
    assert m.HF_ARTIFACTS_REPO == DEFAULT_REPO
    assert m.HF_ARTIFACTS_REVISION == DEFAULT_REVISION


def test_empty_string_falls_back_to_the_default(monkeypatch):
    """The regression. This is what an unset GitHub Actions secret produces."""
    m = _reload(monkeypatch, HF_ARTIFACTS_REPO="", HF_ARTIFACTS_REVISION="")
    assert m.HF_ARTIFACTS_REPO == DEFAULT_REPO, (
        "an empty HF_ARTIFACTS_REPO must mean 'not configured' and fall back; "
        "letting it through sends '' to the Hub, which is the HFValidationError "
        "that killed every scheduled retraining run"
    )
    assert m.HF_ARTIFACTS_REVISION == DEFAULT_REVISION


def test_whitespace_only_is_not_silently_accepted(monkeypatch):
    """A value of " " is a configuration mistake either way.

    It must not reach the Hub as a repo id. Falling back is acceptable;
    carrying the whitespace through is not.
    """
    m = _reload(monkeypatch, HF_ARTIFACTS_REPO="   ")
    assert m.HF_ARTIFACTS_REPO.strip() != "", "a blank repo id must never be used as-is"


def test_a_real_override_still_wins(monkeypatch):
    """The fix must not break the fork workflow the default exists to support."""
    m = _reload(monkeypatch, HF_ARTIFACTS_REPO="someone-else/their-artifacts")
    assert m.HF_ARTIFACTS_REPO == "someone-else/their-artifacts"


@pytest.fixture(autouse=True)
def _restore_module_state(monkeypatch):
    """Leave the module as the rest of the suite expects to find it."""
    yield
    monkeypatch.delenv("HF_ARTIFACTS_REPO", raising=False)
    monkeypatch.delenv("HF_ARTIFACTS_REVISION", raising=False)
    import scripts.artifacts_manifest as manifest

    importlib.reload(manifest)
