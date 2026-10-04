# src/core/tokenizer.py
"""Qwen3 tokenizer wrapper for budget accounting.

THESIS_DECISIONS.md section 2: every token of every call counts toward the budget,
counted with the Qwen tokenizer (not an API counter). The count splits in two,
mirroring the input/output boundary of a call:

  * INPUT (before a call) -- `count_input`: the rendered chat prompt (system +
    re-sent history + the tool schemas) via `apply_chat_template` with
    `add_generation_prompt=True` and Qwen3 thinking enabled. This is exactly the
    string sent to the model, including role tags and special tokens, so raw-text
    counting would undercount. The history is re-sent every call (the model is
    stateless), and that repeated cost is counted on purpose.
  * OUTPUT (after a call) -- the generated completion is split at the LAST
    `</think>` delimiter (self-parsed in `llm_client`; vLLM's chat-mode
    `--reasoning-parser` is not used because we call raw `/v1/completions`).
    Tokens up to and including the delimiter are `thinking_tokens`, the rest
    `answer_tokens`; the tag tokens and inter-part whitespace count as thinking,
    so the two parts sum to the full generated output.

The ledger then checks the identity total = input + thinking + answer (TokenUsage).

AWQ-8B/AWQ-32B parity is verified empirically, not assumed: an automated test asserts
that both served checkpoints render identical chat-template strings AND identical
token-id sequences on five representative prompt shapes currently used by the harness
(see `tests/test_tokenizer.py::test_tokenizer_parity_across_qwen3_sizes`). Chat
templates can change on the Hub independently of the weights, so the tokenizer
`revision` must be pinned (one exact HF commit per checkpoint) before the pilot.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer

__all__ = ["DEFAULT_TOKENIZER", "QwenTokenizer"]

# The tokenizer of the served checkpoint itself (the AWQ repo, not the base repo:
# separate HF repos ship their own tokenizer files and revisions). Parity with
# Qwen3-32B-AWQ is asserted by test, not assumed
# (test_tokenizer_parity_across_qwen3_sizes). AWQ INT4 replaced FP8 because the
# run machine is Ampere (RTX A6000, SM 8.6), which has no native FP8 support --
# see THESIS_DECISIONS.md section 1.
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B-AWQ"


class QwenTokenizer:
    """Thin wrapper over the Hugging Face Qwen3 tokenizer for token counting."""

    def __init__(self, model_name: str = DEFAULT_TOKENIZER, *,
                 revision: str | None = None) -> None:
        """`revision` pins the exact HF commit of the tokenizer files. The pilot and
        the main run MUST set it (chat templates can change on the Hub independently
        of the weights); None (latest) is acceptable for local dev only."""
        self.model_name = model_name
        self.revision = revision
        # Tokenizer files only (no weights); cached after the first load.
        self._tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # `</think>` is a single added token in the Qwen3 vocab; `llm_client`
        # splits generated token ids at its LAST occurrence (thinking | answer).
        self.end_think_token_id: int = self._tok.convert_tokens_to_ids("</think>")

    def count_text(self, text: str) -> int:
        """Token count of a raw string, no special tokens added.

        Used for the OUTPUT halves -- the raw thinking segment (up to and including
        the last `</think>`) and the raw answer segment -- when the endpoint does
        not return generated token ids.
        """
        return len(self._tok.encode(text, add_special_tokens=False))

    def encode_input(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
    ) -> list[int]:
        """Token ids of the prompt actually sent to the model for one call.

        Renders the Qwen3 chat template over `messages` (plus any `tools` schemas)
        with the generation prompt appended, then tokenizes.
        """
        # return_dict=False -> a flat list of token ids. Without it, transformers 5.x
        # returns a BatchEncoding dict (input_ids + attention_mask), whose len() is 2.
        return self._tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=False,
        )

    def count_input(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
    ) -> int:
        """Token count of the prompt actually sent to the model for one call.

        This is the per-call INPUT cost, including the re-sent history.
        """
        return len(self.encode_input(messages, tools=tools, enable_thinking=enable_thinking))

    def render_input(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool = True,
    ) -> str:
        """The rendered prompt string (to send to vLLM, or to inspect/debug)."""
        return self._tok.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            tokenize=False,
        )
