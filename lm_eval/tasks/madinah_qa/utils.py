"""Prompt and answer-label helpers for the MadinahQA OALL v2 tasks."""

from __future__ import annotations

from typing import Any


ARABIC_LABELS = ["أ", "ب", "ج", "د", "هـ"]
LATIN_LABELS = ["A", "B", "C", "D", "E"]
INSTRUCTION = (
    "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الأجوبة:\n\n"
)


def _valid_options(doc: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return available options as ``(Latin label, Arabic label, text)``."""
    options = []
    for index, (latin_label, arabic_label) in enumerate(
        zip(LATIN_LABELS, ARABIC_LABELS), start=1
    ):
        option = doc.get(f"Option {index}")
        if option is not None:
            options.append((latin_label, arabic_label, str(option)))
    if not options:
        raise ValueError("MadinahQA row contains no answer options.")
    return options


def doc_to_text(doc: dict[str, Any]) -> str:
    """Reproduce the Lighteval MadinahQA prompt from the original row."""
    query = (
        f"{INSTRUCTION}\nالسياق:\n{doc.get('Context')}\n"
        f"السؤال:\n{doc['Question']}\n"
    )
    query += "".join(
        f"{arabic_label}. {option}\n"
        for _, arabic_label, option in _valid_options(doc)
    )
    return query + "الإجابة:"


def doc_to_choice(doc: dict[str, Any]) -> list[str]:
    """Return the available Arabic option labels without modifying the row."""
    return [arabic_label for _, arabic_label, _ in _valid_options(doc)]


def doc_to_target(doc: dict[str, Any]) -> str:
    """Return the Arabic label corresponding to the source Latin answer key."""
    answer_key = str(doc["Answer Key"])
    for latin_label, arabic_label, _ in _valid_options(doc):
        if latin_label == answer_key:
            return arabic_label
    raise ValueError(
        f"Answer Key {answer_key!r} is not present in the valid options for "
        f"MadinahQA row {doc.get('ID', '<unknown>')!r}."
    )
