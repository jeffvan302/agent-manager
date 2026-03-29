"""Extractive and LLM-backed summary helpers used by the context pipeline."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agent_manager.types import Message


class SimpleSummarizer:
    """Extractive summarizer that compresses message history into a compact digest.

    Uses sentence scoring based on keyword frequency and positional weighting
    to select the most representative sentences from the conversation.  This
    replaces the earlier stub that simply concatenated the last few messages.
    """

    def __init__(
        self,
        *,
        max_summary_sentences: int = 6,
        min_sentence_length: int = 12,
    ) -> None:
        self.max_summary_sentences = max_summary_sentences
        self.min_sentence_length = min_sentence_length

    def summarize_messages(self, messages: list[Message], limit: int = 20) -> str:
        """Produce a compressed extractive summary of *messages*.

        Parameters
        ----------
        messages:
            The full list of messages to summarise.
        limit:
            Maximum number of recent messages to consider.
        """
        trimmed = messages[-limit:]
        if not trimmed:
            return ""

        # ---- 1. Flatten messages into tagged sentences ----
        tagged_sentences: list[tuple[str, str, int]] = []  # (role, sentence, index)
        for message in trimmed:
            if not message.content:
                continue
            for sentence in self._split_sentences(message.content):
                if len(sentence) >= self.min_sentence_length:
                    tagged_sentences.append(
                        (message.role, sentence, len(tagged_sentences))
                    )

        if not tagged_sentences:
            # Fall back to simple concatenation for very short histories.
            fragments = [
                f"{m.role}: {m.content}" for m in trimmed if m.content
            ]
            return " ".join(fragments)[:600]

        # ---- 2. Score sentences by keyword frequency + position ----
        all_words = Counter(
            word
            for _, sent, _ in tagged_sentences
            for word in self._tokenize(sent)
        )
        total_sentences = len(tagged_sentences)

        scored: list[tuple[float, int, str, str]] = []
        for role, sentence, idx in tagged_sentences:
            words = self._tokenize(sentence)
            if not words:
                continue
            # TF component – favour words that recur across the conversation.
            tf_score = sum(all_words[w] for w in words) / len(words)
            # Position bonus – later sentences are more likely to capture the
            # current state of the conversation.
            position_weight = 0.5 + 0.5 * (idx / max(total_sentences, 1))
            # Role bonus – assistant and user sentences are more informative
            # than system instructions for a summary.
            role_weight = 1.2 if role in ("assistant", "user") else 0.8
            # Tool results carry factual content worth preserving.
            if role == "tool":
                role_weight = 1.1
            score = tf_score * position_weight * role_weight
            scored.append((score, idx, role, sentence))

        # ---- 3. Select top sentences in original order ----
        scored.sort(key=lambda t: t[0], reverse=True)
        selected = scored[: self.max_summary_sentences]
        selected.sort(key=lambda t: t[1])  # restore chronological order

        # ---- 4. Format ----
        parts: list[str] = []
        for _score, _idx, role, sentence in selected:
            parts.append(f"[{role}] {sentence}")

        summary = "\n".join(parts)

        # Ensure the summary is actually shorter than the source material.
        source_length = sum(len(m.content or "") for m in trimmed)
        if len(summary) >= source_length and source_length > 0:
            # Truncate aggressively – this is still better than no compression.
            summary = summary[: int(source_length * 0.6)]

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")

    def _split_sentences(self, text: str) -> list[str]:
        """Rough sentence splitter."""
        raw = self._SENTENCE_RE.split(text.strip())
        return [s.strip() for s in raw if s.strip()]

    _WORD_RE = re.compile(r"[a-z0-9]+")

    def _tokenize(self, text: str) -> list[str]:
        return self._WORD_RE.findall(text.lower())


class ProviderBackedSummarizer:
    """Summarizer that delegates to the LLM for abstractive summaries.

    This requires a provider instance and is used when a model is available
    for context distillation.  It falls back to :class:`SimpleSummarizer`
    when the provider call fails.
    """

    def __init__(
        self,
        provider: Any = None,
        *,
        model: str | None = None,
        max_tokens: int = 300,
        fallback: SimpleSummarizer | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._fallback = fallback or SimpleSummarizer()

    def summarize_messages(self, messages: list[Message], limit: int = 20) -> str:
        """Synchronous wrapper – uses extractive fallback.

        The async version :meth:`summarize_messages_async` will use the model.
        """
        return self._fallback.summarize_messages(messages, limit=limit)

    async def summarize_messages_async(self, messages: list[Message], limit: int = 20) -> str:
        """Use the configured provider to produce an abstractive summary."""
        if self._provider is None:
            return self._fallback.summarize_messages(messages, limit=limit)

        from agent_manager.types import ProviderRequest, Message as Msg

        trimmed = messages[-limit:]
        if not trimmed:
            return ""

        conversation = "\n".join(
            f"{m.role}: {m.content}" for m in trimmed if m.content
        )
        prompt = (
            "Summarize the following conversation into a concise paragraph that "
            "preserves key decisions, facts, tool results, and current goals. "
            "Be brief and factual.\n\n" + conversation
        )

        try:
            request = ProviderRequest(
                model=self._model or getattr(self._provider, "config", None) and self._provider.config.model or "default",
                messages=[Msg(role="user", content=prompt)],
                max_tokens=self._max_tokens,
                temperature=0.2,
            )
            result = await self._provider.generate(request)
            if result.text and len(result.text.strip()) > 20:
                return result.text.strip()
        except Exception:
            pass

        return self._fallback.summarize_messages(messages, limit=limit)
