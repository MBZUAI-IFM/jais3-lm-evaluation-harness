"""Prompt and answer-label helpers for human-translated Arabic MMLU."""

from __future__ import annotations

from typing import Any




def _option_labels(doc: dict[str, Any]) -> list[str]:
    """Return one-based numeric labels for the source choice texts."""
    choices = doc["choices"]
    if not choices:
        raise ValueError("Arabic MMLU-HT row contains no choices.")
    return [str(index) for index in range(1, len(choices) + 1)]


def doc_to_text(doc: dict[str, Any]) -> str:
    """Reproduce the Lighteval prompt without overwriting ``doc['choices']``."""
    labels = _option_labels(doc)
    INSTRUCTION = "السؤال التالي هو سؤال متعدد الإختيارات. اختر الإجابة الصحيحة:\n\n"
    query = f"{INSTRUCTION}{doc['question']}\n"
    query += "".join(
        f"{label}. {choice}\n"
        for label, choice in zip(labels, doc["choices"])
    )
    return query + "الإجابة:"


def doc_to_choice(doc: dict[str, Any]) -> list[str]:
    """Return numeric option labels while preserving the source choice texts."""
    return _option_labels(doc)


def doc_to_target(doc: dict[str, Any]) -> str:
    """Convert the zero-based source answer index to its numeric option label."""
    labels = _option_labels(doc)
    answer_index = int(doc["answer"])
    if not 0 <= answer_index < len(labels):
        raise ValueError(
            f"Gold index {answer_index} is outside the {len(labels)} available choices."
        )
    return labels[answer_index]
