# tests/test_live_smoke_cli.py
"""The live-smoke CLI wiring, offline: no endpoint, no tokenizer download.

Only one thing here really needs locking down -- that `--max-model-len` reaches the
client. It is the difference between a smoke run that exercises the context guard and
one that silently does not, and nothing downstream would reveal the omission: a run with
no clamping looks exactly like a run whose window was never applied.
"""

from __future__ import annotations

import pytest

import scripts.run_live_smoke as smoke


class _RecordingClient:
    """Stands in for LLMClient and remembers how it was constructed."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs


def _patch_client(monkeypatch) -> None:
    """Intercept the imports `main` does lazily, so nothing touches the network."""
    import src.core as core

    monkeypatch.setattr(core, "LLMClient", _RecordingClient)
    monkeypatch.setattr(core, "QwenTokenizer", lambda *a, **k: object())

    def _stop_before_running(**kwargs):
        raise RuntimeError("stop: the client has been built, which is all this checks")

    import src.harness as harness
    monkeypatch.setattr(harness, "run_single_instance", _stop_before_running)


def test_max_model_len_reaches_the_client(monkeypatch) -> None:
    _patch_client(monkeypatch)
    with pytest.raises(RuntimeError, match="stop: the client has been built"):
        smoke.main(["--max-model-len", "4096", "--cap", "8000", "--n-runs", "1"])
    assert _RecordingClient.last_kwargs["max_model_len"] == 4096


def test_the_default_window_is_the_served_one(monkeypatch) -> None:
    _patch_client(monkeypatch)
    with pytest.raises(RuntimeError):
        smoke.main(["--cap", "8000", "--n-runs", "1"])
    assert _RecordingClient.last_kwargs["max_model_len"] == 32768


def test_a_nonpositive_window_is_refused_before_anything_is_built(monkeypatch) -> None:
    _patch_client(monkeypatch)
    _RecordingClient.last_kwargs = {}
    with pytest.raises(SystemExit, match="positive number of tokens"):
        smoke.main(["--max-model-len", "0"])
    assert _RecordingClient.last_kwargs == {}  # never got as far as a client
