# SPDX-License-Identifier: Apache-2.0
"""
Thinking/reasoning content parser for separating <think>...</think> blocks.

Provides both streaming (ThinkingParser) and non-streaming (extract_thinking)
interfaces for separating reasoning content from regular response content.

Used by reasoning models like DeepSeek R1, Qwen3/3.5, MiniMax that wrap
their chain-of-thought reasoning in <think>...</think> tags.
"""

import re
from typing import Tuple


# Tags used for thinking blocks
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_OPEN_LEN = len(_OPEN_TAG)   # 7
_CLOSE_LEN = len(_CLOSE_TAG)  # 8

# Regex for non-streaming extraction (complete text)
_THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Handle case where <think> is missing but </think> is present
# (scheduler prepends <think>\n but the tag may be split)
_THINKING_TAIL_PATTERN = re.compile(r'^(.*?)</think>', re.DOTALL)


def extract_thinking(text: str) -> Tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: ``<think>reasoning</think>answer`` → ``("reasoning", "answer")``
    - No thinking: ``just answer`` → ``("", "just answer")``
    - Partial (no open tag): ``reasoning</think>answer`` → ``("reasoning", "answer")``
    - Empty think: ``<think></think>answer`` → ``("", "answer")``
    - Think only: ``<think>reasoning</think>`` → ``("reasoning", "")``

    Args:
        text: Complete model output text.

    Returns:
        Tuple of (thinking_content, regular_content).
    """
    if not text:
        return ("", "")

    thinking_parts = []
    remaining = text

    # Extract all <think>...</think> blocks
    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]

    if thinking_parts:
        # After extracting <think>...</think> blocks, check remaining for
        # stray </think> tags (leaked reasoning from forced budget closure).
        # Text before a stray </think> is additional thinking content.
        while '</think>' in remaining:
            match = _THINKING_TAIL_PATTERN.match(remaining.lstrip())
            if match:
                leaked = remaining.lstrip()[:match.end()]
                thinking_parts.append(match.group(1).strip())
                remaining = remaining.lstrip()[match.end():]
            else:
                break
        thinking = "\n".join(p for p in thinking_parts if p).strip()
        return (thinking, remaining.strip())

    # Handle partial: content before </think> without <think> tag
    if '</think>' in text and '<think>' not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            return (thinking, remaining)

    return ("", text)


class ThinkingParser:
    """Stateful streaming parser for separating <think>...</think> from content.

    Handles streaming chunks where tags may span multiple chunks.
    Returns (thinking_delta, content_delta) tuples for each feed() call.

    Example::

        parser = ThinkingParser()

        # Chunk 1: "<think>Let me"
        t, c = parser.feed("<think>Let me")
        # t = "Let me", c = ""

        # Chunk 2: " think</think>Answer"
        t, c = parser.feed(" think</think>Answer")
        # t = " think", c = "Answer"

        # Flush remaining
        t, c = parser.finish()
    """

    def __init__(self):
        self._in_thinking: bool = False
        self._buffer: str = ""  # Buffer for potential partial tags

    def feed(self, text: str) -> Tuple[str, str]:
        """Feed a text chunk, return (thinking_delta, content_delta).

        Args:
            text: New text chunk from model output.

        Returns:
            Tuple of (thinking_text, content_text) extracted from this chunk.
        """
        if not text:
            return ("", "")

        # Prepend any buffered partial tag content
        text = self._buffer + text
        self._buffer = ""

        thinking_out = []
        content_out = []

        i = 0
        while i < len(text):
            if text[i] == '<':
                # Check if this could be a tag start
                remaining = text[i:]

                # Try to match <think>
                if remaining.startswith(_OPEN_TAG):
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue

                # Try to match </think>
                if remaining.startswith(_CLOSE_TAG):
                    if not self._in_thinking:
                        # Stray </think> while not in thinking mode — leaked
                        # reasoning from forced budget closure. Reclassify
                        # content accumulated in THIS feed() call as thinking.
                        thinking_out.extend(content_out)
                        content_out = []
                    self._in_thinking = False
                    i += _CLOSE_LEN
                    continue

                # Check if it could be a partial tag (not enough chars yet)
                if self._could_be_tag(remaining):
                    # Buffer the rest and wait for more data
                    self._buffer = remaining
                    break

                # Not a tag, emit the '<' as regular content
                if self._in_thinking:
                    thinking_out.append('<')
                else:
                    content_out.append('<')
                i += 1
            else:
                if self._in_thinking:
                    thinking_out.append(text[i])
                else:
                    content_out.append(text[i])
                i += 1

        return ("".join(thinking_out), "".join(content_out))

    def finish(self) -> Tuple[str, str]:
        """Flush any remaining buffered content.

        Should be called when the stream is complete to emit any
        buffered characters that were waiting for potential tag completion.

        Returns:
            Tuple of (thinking_text, content_text) from remaining buffer.
        """
        if not self._buffer:
            return ("", "")

        # The buffer contains a partial tag that never completed.
        # Emit it as-is in the current mode.
        buf = self._buffer
        self._buffer = ""

        if self._in_thinking:
            return (buf, "")
        else:
            return ("", buf)

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        """Check if text could be the start of a <think> or </think> tag.

        Returns True if text is a proper prefix of either tag but not
        yet a complete match.
        """
        length = len(text)
        if length >= _CLOSE_LEN:
            # Long enough to determine - not a partial tag
            return False

        # Check against both tags
        if _OPEN_TAG[:length] == text:
            return True
        if _CLOSE_TAG[:length] == text:
            return True

        return False


class ThinkingBudgetProcessor:
    """Logits processor that enforces min/max thinking token budgets.

    Tracks whether the model is inside a <think>...</think> block by scanning
    the token sequence for think_start_id and think_end_id.

    - Below budget_min: sets </think> logit to -inf (prevents early closure)
    - At/above budget_max: sets all logits except </think> to -inf (forces closure)
    - Outside a thinking block or between min and max: no-op

    Signature matches mlx-lm logits processor convention:
        processor(tokens: mx.array, logits: mx.array) -> mx.array
    """

    def __init__(
        self,
        think_start_id: int,
        think_end_id: int,
        budget_min: int | None = None,
        budget_max: int | None = None,
    ):
        self.think_start_id = think_start_id
        self.think_end_id = think_end_id
        self.budget_min = budget_min
        self.budget_max = budget_max

    def __call__(self, tokens: "mx.array", logits: "mx.array") -> "mx.array":
        import mlx.core as mx

        if self.budget_min is None and self.budget_max is None:
            return logits

        # Find the last <think> and </think> positions in the token sequence
        token_list = tokens.tolist()
        last_think_start = -1
        last_think_end = -1
        for i, t in enumerate(token_list):
            if t == self.think_start_id:
                last_think_start = i
            elif t == self.think_end_id:
                last_think_end = i

        # Not in a thinking block: no-op
        if last_think_start < 0 or last_think_end > last_think_start:
            return logits

        # Count tokens generated since <think> (excluding the <think> token itself)
        thinking_tokens = len(token_list) - last_think_start - 1

        # Below min: suppress </think> to prevent early closure
        if self.budget_min is not None and thinking_tokens < self.budget_min:
            logits = mx.array(logits)  # ensure writable copy
            logits[0, self.think_end_id] = float('-inf')
            return logits

        # At/above max: force </think> by zeroing everything else
        if self.budget_max is not None and thinking_tokens >= self.budget_max:
            forced = mx.full(logits.shape, float('-inf'))
            forced[0, self.think_end_id] = logits[0, self.think_end_id]
            return forced

        return logits
