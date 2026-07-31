# tests/test_usage_audit.py
"""Endpoint-usage audit layer tests (all offline, no endpoint, no network).

What must hold:
  * LLMResponse accepts the optional endpoint fields and old construction works;
  * extraction handles OpenAI-style objects, dicts, nested details, and absence;
  * comparison marks missing endpoint usage honestly (null fields, never faked);
  * the harness JSON gains usage_audit + transcript_sanity blocks, and an
    offline scripted client yields has_endpoint_usage = false;
  * the live-smoke CLI imports without touching any endpoint;
  * the transcript sanity helper flags <think> leaks and stays quiet on clean runs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.core import LLMClient, LLMResponse, audit_run, compare_call, extract_endpoint_usage
from src.harness import run_single_instance, transcript_sanity
from src.schemas import (
    CallRecord,
    GeneratorParams,
    Instance,
    Person,
    TokenUsage,
    TravelStructure,
)


# --------------------------------------------------------------------------- #
# 1. LLMResponse: optional endpoint fields, backward compatible.
# --------------------------------------------------------------------------- #
def test_llm_response_accepts_optional_endpoint_usage() -> None:
    # Old construction style (all existing scripted clients) still works.
    old = LLMResponse(
        thinking="", answer="ok", raw_text="ok",
        input_tokens=10, thinking_tokens=0, answer_tokens=1, finish_reason="stop",
    )
    assert old.endpoint_total_tokens is None
    assert old.endpoint_usage_raw is None
    assert old.usage_source is None

    new = LLMResponse(
        thinking="r", answer="a", raw_text="<think>r</think>a",
        input_tokens=10, thinking_tokens=1, answer_tokens=1, finish_reason="stop",
        endpoint_prompt_tokens=12, endpoint_completion_tokens=5,
        endpoint_total_tokens=17, endpoint_reasoning_tokens=3,
        endpoint_cached_tokens=0, endpoint_usage_raw={"prompt_tokens": 12},
        endpoint_model="qwen-test", usage_source="endpoint.usage",
    )
    assert new.endpoint_total_tokens == 17
    assert new.endpoint_reasoning_tokens == 3


# --------------------------------------------------------------------------- #
# 2 + 3. Comparison with and without endpoint usage.
# --------------------------------------------------------------------------- #
def test_usage_audit_missing_endpoint_usage() -> None:
    result = compare_call(
        internal_input_tokens=100, internal_thinking_tokens=40,
        internal_answer_tokens=10, endpoint=None, role="react_step",
    )
    assert result["has_endpoint_usage"] is False
    assert result["internal_total_tokens"] == 150
    assert result["endpoint_prompt_tokens"] is None
    assert result["endpoint_total_tokens"] is None
    assert result["delta_total_tokens"] is None
    assert result["relative_delta_total"] is None
    assert any("missing" in n for n in result["notes"])

    # Whole-run audit with a client that logs no usage at all (offline scripted).
    tokens = TokenUsage(cap=1000, total=150, input_tokens=100, thinking_tokens=40,
                        answer_tokens=10, n_calls=1, budget_exhausted=False)
    calls = [CallRecord(role="react_step", input_tokens=100, thinking_tokens=40,
                        answer_tokens=10, latency_seconds=0.0)]
    audit = audit_run(tokens=tokens, calls=calls, endpoint_records=None)
    assert audit["has_endpoint_usage"] is False
    assert audit["totals"]["endpoint_total_tokens"] is None
    assert audit["totals"]["delta_total_tokens"] is None
    assert json.dumps(audit)  # JSON-serialisable


def test_usage_audit_with_endpoint_usage() -> None:
    endpoint = {"prompt_tokens": 105, "completion_tokens": 55, "total_tokens": 160,
                "reasoning_tokens": 40, "cached_tokens": None}
    result = compare_call(
        internal_input_tokens=100, internal_thinking_tokens=40,
        internal_answer_tokens=10, endpoint=endpoint, role="react_step",
    )
    assert result["has_endpoint_usage"] is True
    assert result["endpoint_total_tokens"] == 160
    assert result["delta_total_tokens"] == 10          # 160 endpoint - 150 internal
    assert result["relative_delta_total"] == 10 / 150
    assert result["endpoint_reasoning_tokens"] == 40

    # Run-level totals across two aligned calls.
    tokens = TokenUsage(cap=4000, total=300, input_tokens=200, thinking_tokens=80,
                        answer_tokens=20, n_calls=2, budget_exhausted=False)
    calls = [
        CallRecord(role="react_step", input_tokens=100, thinking_tokens=40,
                   answer_tokens=10, latency_seconds=0.0),
        CallRecord(role="finalize", input_tokens=100, thinking_tokens=40,
                   answer_tokens=10, latency_seconds=0.0),
    ]
    records = [
        {"prompt_tokens": 105, "completion_tokens": 55, "total_tokens": 160},
        {"prompt_tokens": 102, "completion_tokens": 52, "total_tokens": 154},
    ]
    audit = audit_run(tokens=tokens, calls=calls, endpoint_records=records)
    assert audit["has_endpoint_usage"] is True
    assert audit["calls_aligned"] is True
    assert audit["totals"]["endpoint_total_tokens"] == 314
    assert audit["totals"]["delta_total_tokens"] == 14
    assert audit["per_call"][0]["endpoint_total_tokens"] == 160
    assert json.dumps(audit)


def test_usage_audit_reconstructs_total_when_missing() -> None:
    endpoint = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": None}
    result = compare_call(
        internal_input_tokens=100, internal_thinking_tokens=0,
        internal_answer_tokens=45, endpoint=endpoint,
    )
    assert result["endpoint_total_tokens"] == 150
    assert result["delta_total_tokens"] == 5
    assert any("reconstructed" in n for n in result["notes"])


# --------------------------------------------------------------------------- #
# 5. Extraction from OpenAI-style shapes (pure + through LLMClient, no network).
# --------------------------------------------------------------------------- #
def test_extract_endpoint_usage_shapes() -> None:
    assert extract_endpoint_usage(None) is None

    as_dict = extract_endpoint_usage({
        "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
        "completion_tokens_details": {"reasoning_tokens": 5},
        "prompt_tokens_details": {"cached_tokens": 2},
    })
    assert as_dict is not None
    assert (as_dict["prompt_tokens"], as_dict["completion_tokens"],
            as_dict["total_tokens"]) == (11, 7, 18)
    assert as_dict["reasoning_tokens"] == 5 and as_dict["cached_tokens"] == 2

    as_obj = extract_endpoint_usage(SimpleNamespace(
        prompt_tokens=20, completion_tokens=9, total_tokens=29,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=6),
    ))
    assert as_obj is not None and as_obj["total_tokens"] == 29
    assert as_obj["reasoning_tokens"] == 6
    assert json.dumps(as_obj["raw"])  # raw form is JSON-serialisable

    # An object with no recognisable counts is treated as no usage (honest).
    assert extract_endpoint_usage(SimpleNamespace(unrelated=1)) is None


class _StubTokenizer:
    """Duck-typed stand-in so LLMClient needs no HF download in this test."""

    def render_input(self, messages: list[dict[str, Any]], *, tools: Any = None,
                     enable_thinking: bool = True) -> str:
        return " ".join(m["content"] for m in messages)

    def count_text(self, text: str) -> int:
        return len(text.split())


def test_llm_client_parses_openai_style_usage() -> None:
    client = LLMClient(tokenizer=_StubTokenizer(), model="qwen-test",
                       base_url="http://localhost:9/v1")  # never contacted
    fake_completion = SimpleNamespace(
        choices=[SimpleNamespace(text="<think>plan it</think> Action: finish",
                                 finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        ),
        model="qwen-test-0.1",
    )
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return fake_completion

    client._client = SimpleNamespace(
        completions=SimpleNamespace(create=fake_create)
    )

    resp = client.complete([{"role": "user", "content": "go"}], max_tokens=64)

    assert calls, "the fake endpoint was not called"
    assert resp.thinking == "plan it" and resp.answer == "Action: finish"
    assert resp.endpoint_prompt_tokens == 100
    assert resp.endpoint_completion_tokens == 50
    assert resp.endpoint_total_tokens == 150
    assert resp.endpoint_reasoning_tokens == 30
    assert resp.endpoint_model == "qwen-test-0.1"
    assert resp.usage_source == "endpoint.usage"
    # The audit trail got exactly one aligned record.
    assert len(client.usage_log) == 1
    assert client.usage_log[0]["total_tokens"] == 150
    # Internal counts still come from the tokenizer, not the endpoint.
    assert resp.thinking_tokens == 2 and resp.answer_tokens == 2


def test_llm_client_records_none_when_usage_absent() -> None:
    client = LLMClient(tokenizer=_StubTokenizer(), model="qwen-test",
                       base_url="http://localhost:9/v1")
    fake_completion = SimpleNamespace(
        choices=[SimpleNamespace(text="ok", finish_reason="stop")], usage=None,
    )
    client._client = SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: fake_completion)
    )
    resp = client.complete([{"role": "user", "content": "go"}], max_tokens=8)
    assert resp.endpoint_total_tokens is None and resp.usage_source is None
    assert client.usage_log == [None]  # call happened, endpoint reported nothing


# --------------------------------------------------------------------------- #
# 4. Harness JSON carries the audit blocks; offline client fakes nothing.
# --------------------------------------------------------------------------- #
def _instance() -> Instance:
    return Instance(
        instance_id="usage-audit-test",
        seed=0,
        generator_params=GeneratorParams(
            n_people=2, tightness=0.5, overlap=0.5,
            travel_structure=TravelStructure.UNIFORM,
        ),
        start_location="S",
        start_time=0,
        end_of_day=300,
        meeting_duration=30,
        waiting_allowed=True,
        tie_break="earliest_start",
        locations=["S", "A", "B"],
        people=[
            Person(person_id="p0", location="A", window_start=0, window_end=100),
            Person(person_id="p1", location="B", window_start=0, window_end=200),
        ],
        travel_times={
            "S": {"S": 0, "A": 10, "B": 10},
            "A": {"S": 10, "A": 0, "B": 20},
            "B": {"S": 10, "A": 20, "B": 0},
        },
    )


class _ScriptedClient:
    """Offline scripted client WITHOUT a usage_log (like OfflineSmokeClient)."""

    def __init__(self, answers: list[str]):
        self._answers = answers
        self.i = 0

    def count_input(self, messages: list[dict[str, Any]], *, enable_thinking: bool = True,
                    tools: Any = None) -> int:
        return 50

    def complete(self, messages: list[dict[str, Any]], *, max_tokens: int,
                 enable_thinking: bool = True, tools: Any = None,
                 guided_json: Any = None, stop: Any = None) -> LLMResponse:
        text = self._answers[self.i]
        self.i += 1
        return LLMResponse(
            thinking="", answer=text, raw_text=text,
            input_tokens=50, thinking_tokens=0,
            answer_tokens=len(text.split()), finish_reason="stop",
        )


class _ScriptedClientWithUsage(_ScriptedClient):
    """Same, but exposing an LLMClient-style usage_log (simulates a live client)."""

    def __init__(self, answers: list[str]):
        super().__init__(answers)
        self.usage_log: list[dict[str, Any] | None] = []

    def complete(self, *args: Any, **kwargs: Any) -> LLMResponse:
        resp = super().complete(*args, **kwargs)
        internal = resp.input_tokens + resp.thinking_tokens + resp.answer_tokens
        self.usage_log.append({
            "prompt_tokens": resp.input_tokens + 2,       # endpoint counts a bit more
            "completion_tokens": resp.answer_tokens + 1,  # (tags/whitespace) -- fake
            "total_tokens": internal + 3,                 # but consistent for the test
            "reasoning_tokens": None, "cached_tokens": None,
            "model": "fake-live", "finish_reason": "stop", "raw": {},
        })
        return resp


_SCRIPT = [
    "Action: propose[p0@10]",
    "Action: finish",
    '{"meetings": [{"person_id": "p0", "start_time": 10}]}',
]


def test_result_writer_includes_usage_audit(tmp_path) -> None:
    run = run_single_instance(
        instance=_instance(), client=_ScriptedClient(list(_SCRIPT)),
        condition="c1_react", cap=4000, model_label="scripted-test",
        output_dir=tmp_path,
    )
    doc = json.loads(run.json_path.read_text(encoding="utf-8"))

    audit = doc["usage_audit"]
    # Offline scripted client: endpoint usage honestly absent, nothing faked.
    assert audit["has_endpoint_usage"] is False
    assert audit["n_endpoint_records"] is None
    assert audit["totals"]["endpoint_total_tokens"] is None
    assert audit["totals"]["delta_total_tokens"] is None
    assert audit["totals"]["internal_total_tokens"] == doc["run_result"]["tokens"]["total"]
    assert len(audit["per_call"]) == doc["run_result"]["tokens"]["n_calls"]

    sanity = doc["transcript_sanity"]
    assert sanity["think_leak"] is False
    assert sanity["clean"] is True


def test_harness_usage_audit_with_live_style_client(tmp_path) -> None:
    client = _ScriptedClientWithUsage(list(_SCRIPT))
    run = run_single_instance(
        instance=_instance(), client=client,
        condition="c1_react", cap=4000, model_label="fake-live",
        output_dir=tmp_path,
    )
    audit = run.usage_audit
    assert audit["has_endpoint_usage"] is True
    assert audit["calls_aligned"] is True
    internal_total = run.run_result.tokens.total
    assert audit["totals"]["endpoint_total_tokens"] == internal_total + 3 * 3
    assert audit["totals"]["delta_total_tokens"] == 9
    # A reused client only contributes THIS run's slice.
    client._answers, client.i = list(_SCRIPT), 0
    run2 = run_single_instance(
        instance=_instance(), client=client,
        condition="c1_react", cap=4000, model_label="fake-live",
    )
    assert run2.usage_audit["n_endpoint_records"] == 3  # not 6


# --------------------------------------------------------------------------- #
# 6. Live-smoke CLI imports safely (no endpoint, no tokenizer download).
# --------------------------------------------------------------------------- #
def test_live_smoke_script_imports_and_argparse() -> None:
    from scripts.run_live_smoke import BANNER, build_parser

    assert "NOT PILOT" in BANNER
    args = build_parser().parse_args([])
    assert args.condition == "c1_react"
    assert args.cap == 2000
    assert args.n_runs == 1
    args2 = build_parser().parse_args(
        ["--condition", "c2_verify_revise", "--require-endpoint-usage",
         "--base-url", "http://localhost:8000/v1"]
    )
    assert args2.require_endpoint_usage is True
    assert args2.base_url == "http://localhost:8000/v1"


# --------------------------------------------------------------------------- #
# 7. Transcript sanity diagnostics.
# --------------------------------------------------------------------------- #
def test_transcript_sanity_detects_think_leak() -> None:
    leaked = [
        {"role": "system", "content": "You are a meeting-planning agent."},
        {"role": "assistant", "content": "<think>I should hide this</think> plan"},
    ]
    diag = transcript_sanity(leaked)
    assert diag["think_leak"] is True
    assert diag["think_open_message_indices"] == [1]
    assert diag["clean"] is False

    clean = [
        {"role": "system", "content": "You are a meeting-planning agent."},
        {"role": "assistant", "content": "Action: finish"},
    ]
    diag2 = transcript_sanity(clean)
    assert diag2["think_leak"] is False
    assert diag2["oracle_term_hits"] == []
    assert diag2["clean"] is True


def test_transcript_sanity_flags_oracle_terms() -> None:
    transcript = [
        {"role": "user", "content": "Observation: the solver optimum is 4."},
    ]
    diag = transcript_sanity(transcript)
    terms = {hit["term"] for hit in diag["oracle_term_hits"]}
    assert {"solver", "optimum"} <= terms
    assert diag["clean"] is False
