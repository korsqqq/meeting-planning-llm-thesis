# tests/test_tokenizer.py
"""QwenTokenizer smoke tests.

Loads the real Qwen3 tokenizer (cached after first download). If it cannot load --
no network on a fresh machine -- the tests skip rather than fail, since the budget
LEDGER tests (test_budget.py) cover the accounting logic offline.

EXCEPTION: with `REQUIRE_TOKENIZER_PARITY=1` a load failure FAILS instead of
skipping. The pilot checklist runs the suite with this flag set and archives the
`... passed, 0 skipped` output -- otherwise "all green" could silently mean the
parity evidence never ran (THESIS_DECISIONS section 2).
"""

from __future__ import annotations

import os

import pytest

from src.core import QwenTokenizer


def _load_or_skip(builder, what: str):
    """Build tokenizer(s); on failure skip -- or FAIL under REQUIRE_TOKENIZER_PARITY."""
    try:
        return builder()
    except Exception as exc:  # network / cache miss on a fresh machine
        if os.environ.get("REQUIRE_TOKENIZER_PARITY"):
            pytest.fail(f"required tokenizer test could not run ({what}): {exc}")
        pytest.skip(f"{what} unavailable: {exc}")


@pytest.fixture(scope="module")
def tok() -> QwenTokenizer:
    return _load_or_skip(QwenTokenizer, "Qwen tokenizer")


def test_count_text_basic(tok: QwenTokenizer) -> None:
    # "Hello world" is two Qwen tokens; no special tokens are added.
    assert tok.count_text("Hello world") == 2
    assert tok.count_text("") == 0


def test_count_text_deterministic(tok: QwenTokenizer) -> None:
    text = "Plan a meeting with Alice at 10:00, then Bob at 11:30."
    assert tok.count_text(text) == tok.count_text(text)


def test_count_input_includes_chat_scaffold(tok: QwenTokenizer) -> None:
    content = "Hello"
    messages = [{"role": "user", "content": content}]
    # The rendered prompt carries role tags + special tokens + generation prompt, so
    # it must exceed the raw content by several tokens. (A strict ">" is too weak --
    # it once passed at 2 when a bug returned the dict key count instead of tokens.)
    assert tok.count_input(messages) >= tok.count_text(content) + 5


def test_tool_schemas_add_input_tokens(tok: QwenTokenizer) -> None:
    messages = [{"role": "user", "content": "Who can I meet?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_people",
                "description": "List the person_ids in the instance.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert tok.count_input(messages, tools=tools) > tok.count_input(messages)


def test_render_input_is_a_string(tok: QwenTokenizer) -> None:
    rendered = tok.render_input([{"role": "user", "content": "Hi"}])
    assert isinstance(rendered, str) and len(rendered) > 0


# --------------------------------------------------------------------------- #
# FP8-8B <-> FP8-32B tokenizer parity, on the EXACT repos the thesis serves
# (`Qwen/Qwen3-*-FP8` are separate HF repos with their own tokenizer files and
# revisions -- base-repo parity would not prove anything about them).
#
# Caps calibrated on the 8B pilot transfer to the 32B main run ONLY IF both
# checkpoints render identical chat-template strings and produce identical
# token-id sequences on the prompts the harness sends. That is an empirical
# claim, not a Qwen guarantee (chat templates on the Hub can change
# independently of the weights), so it is asserted here over five
# representative prompt shapes currently used by the harness: plain
# system+user, tool schemas, a ReAct Thought/Action/Observation history,
# the thinking-OFF finalise prompt, and non-ASCII content.
# --------------------------------------------------------------------------- #
PARITY_TOKENIZER_8B = "Qwen/Qwen3-8B-FP8"
PARITY_TOKENIZER_32B = "Qwen/Qwen3-32B-FP8"
_PARITY_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_travel_time",
        "description": "Travel minutes between two locations.",
        "parameters": {
            "type": "object",
            "properties": {"from_loc": {"type": "string"}, "to_loc": {"type": "string"}},
            "required": ["from_loc", "to_loc"],
        },
    },
}]

_PARITY_CASES: dict[str, dict] = {
    "plain": {
        "messages": [
            {"role": "system", "content": "You are a meeting planner."},
            {"role": "user", "content": "Plan meetings with p0..p8 under the budget."},
        ],
    },
    "tool_schemas": {
        "messages": [
            {"role": "system", "content": "You are a meeting planner."},
            {"role": "user", "content": "Who can I meet first?"},
        ],
        "tools": _PARITY_TOOLS,
    },
    # The shape react_core actually builds: system prompt with tool docs as text,
    # then alternating assistant (Thought/Action) and user (Observation:) turns.
    "react_history": {
        "messages": [
            {"role": "system", "content": "You are a meeting-planning agent. "
             "Act with one line: Action: <tool>[args] or Action: finish."},
            {"role": "user", "content": "Plan the route now. Start by listing the people."},
            {"role": "assistant", "content": "Thought: I need the people first.\n"
             "Action: list_people[]"},
            {"role": "user", "content": "Observation: p0, p1, p2"},
            {"role": "assistant", "content": "Thought: check p0's window.\n"
             "Action: get_availability[p0]"},
            {"role": "user", "content": "Observation: p0 available 10-60 at loc_a"},
        ],
    },
    # The terminal serialise prompt (finalize_step): thinking OFF.
    "finalize_thinking_off": {
        "messages": [
            {"role": "system", "content": "Serialise the given meeting plan as JSON. "
             "Do not change it."},
            {"role": "user", "content": "Plan: p0@10, p1@60\nReturn the JSON now."},
        ],
        "enable_thinking": False,
    },
    "non_ascii": {
        "messages": [
            {"role": "system", "content": "You are a meeting planner."},
            {"role": "user", "content": "Plane Treffen mit Björn (Café Zürich), "
             "Алисой und Бобом — 10:00, 11:30; предпочтения: «окно» ≤ 45 мин."},
        ],
    },
}


@pytest.fixture(scope="module")
def tok_pair() -> tuple[QwenTokenizer, QwenTokenizer]:
    from src.core import DEFAULT_TOKENIZER

    # The budget-counting tokenizer must be the served 8B checkpoint itself.
    assert DEFAULT_TOKENIZER == PARITY_TOKENIZER_8B
    return _load_or_skip(
        lambda: (QwenTokenizer(PARITY_TOKENIZER_8B), QwenTokenizer(PARITY_TOKENIZER_32B)),
        "an FP8 Qwen tokenizer pair",
    )


@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_tokenizer_parity_across_qwen3_sizes(
    tok_pair: tuple[QwenTokenizer, QwenTokenizer], case: str
) -> None:
    tok8, tok32 = tok_pair
    spec = _PARITY_CASES[case]
    kwargs = {
        "tools": spec.get("tools"),
        "enable_thinking": spec.get("enable_thinking", True),
    }
    rendered8 = tok8.render_input(spec["messages"], **kwargs)
    rendered32 = tok32.render_input(spec["messages"], **kwargs)
    assert rendered8 == rendered32, f"[{case}] rendered chat-template strings differ"

    ids8 = tok8.encode_input(spec["messages"], **kwargs)
    ids32 = tok32.encode_input(spec["messages"], **kwargs)
    assert ids8 == ids32, (
        f"[{case}] token-id sequences differ: 8B={len(ids8)} vs 32B={len(ids32)} tokens"
    )
    # count_input is the budget-facing view of the same ids.
    assert tok8.count_input(spec["messages"], **kwargs) == len(ids8)
