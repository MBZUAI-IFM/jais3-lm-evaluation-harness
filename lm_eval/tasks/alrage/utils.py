"""Prompting and exact-match scoring for the ALRAGE OALL v2 task."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_ARABIC_DIACRITICS_AND_TATWEEL = re.compile(
    r"[\u0610-\u061a\u0640\u064b-\u065f\u0670\u06d6-\u06ed]"
)


def parse_candidates(candidates: Any) -> list[str]:
    """Parse candidates supplied as a list or a newline-separated string."""
    if isinstance(candidates, (list, tuple)):
        parsed = [str(candidate).strip() for candidate in candidates]
    else:
        parsed = [candidate.strip() for candidate in str(candidates).splitlines()]
    parsed = [candidate for candidate in parsed if candidate]
    if not parsed:
        raise ValueError("ALRAGE row contains no non-empty candidate contexts.")
    return parsed


def doc_to_text(doc: dict[str, Any]) -> str:
    """Reproduce the source Lighteval ALRAGE prompt."""
    candidates = parse_candidates(doc["candidates"])
    INSTRUCTION = "بناءً على السياقات المقترحة التالية، اجب عن السؤال التالي"
    return (
        f"{INSTRUCTION}\n\nالسؤال:\n{doc['question']}\n\n"
        f"السياقات المقترحة:\n{', '.join(candidates)}\n"
    )


def normalize_arabic_answer(text: Any) -> str:
    """Normalize Arabic free-form answers for deterministic comparison."""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    normalized = _ARABIC_DIACRITICS_AND_TATWEEL.sub("", normalized)
    normalized = re.sub(r"[إأآا]", "ا", normalized)
    normalized = normalized.replace("ى", "ي").replace("ة", "ه")
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def process_results_exact_match(doc: dict[str, Any], results: list[str]):
    """Return Arabic-normalized exact match for one generated answer."""
    prediction = results[0] if results else ""
    reference = doc["gold_answer"]
    return {
        "exact_match": float(
            normalize_arabic_answer(prediction)
            == normalize_arabic_answer(reference)
        )
    }
