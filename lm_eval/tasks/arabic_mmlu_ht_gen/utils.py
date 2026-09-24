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
    header_instruction = build_prompt(instruction_lang="ar", labels_lang="num_en", num_choices=len(labels))
    return header_instruction + "\n" + query + "الإجابة:"


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


import re
BRACKETED_LABEL_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")


def normalize_label(label):
    return str(label).strip().lower()


def extract_bracketed_label(resps, docs):
    def extract(resp):
        if not isinstance(resp, str):
            return "[invalid]"

        matches = BRACKETED_LABEL_RE.findall(resp)
        if not matches:
            return "[invalid]"

        return matches[-1]

    return map(lambda r: extract(r[0] if r else ""), resps)


def exact_match_normalized_label(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else "[invalid]"
    reference = references[0] if references else "[invalid]"
    return {
        "exact_match": float(
            normalize_label(prediction) == normalize_label(reference)
        )
    }


def build_prompt(instruction_lang: str, labels_lang: str, num_choices: int) -> str:
    if instruction_lang not in ["en", "ar"]:
        raise ValueError("instruction_lang must be 'en' or 'ar'")
    if labels_lang not in ["en", "ar", "num_en"]:
        raise ValueError("labels_lang must be 'en' or 'ar' or 'num_en'")
    if num_choices < 2 or num_choices > 26:
        raise ValueError("num_choices must be between 2 and 26")

    # Generate letters
    letters_en = [chr(ord('A') + i) for i in range(num_choices)]
    letters_ar = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي",
                  "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر",
                  "ش", "ت", "ث", "خ", "ذ", "ض"][:num_choices]
    letters_num_en = [str(i) for i in range(1, num_choices + 1)]

    # Choose label set
    letters = letters_en if labels_lang == "en" else letters_ar if labels_lang == "ar" else letters_num_en

    # Instruction text
    if instruction_lang == "en":
        instruction = "# Instruction\nReturn only the final answer as the option letter inside double square brackets."
        sep = ", "
        or_word = "or"
        valid_prefix = "Valid outputs are only:\n"
    else:
        instruction = "# التعليمات\nأعد فقط الإجابة النهائية على شكل حرف الخيار داخل أقواس مربعة مزدوجة."
        sep = "، "
        or_word = "أو"
        valid_prefix = "المخرجات المسموح بها فقط هي:\n"

    # Build options string
    if num_choices == 2:
        options = f"[[{letters[0]}]] {or_word} [[{letters[1]}]]"
    else:
        options = sep.join(f"[[{l}]]" for l in letters[:-1])
        options += f"{sep}{or_word} [[{letters[-1]}]]"

    valid_line = f"{valid_prefix}{options}"

    return f"{instruction}\n{valid_line}"


def add_options_with_text(dataset):
    def _add_field(doc):
        labels = _option_labels(doc)
        options_text = "".join(
            f"{label}. {choice}\n"
            for label, choice in zip(labels, doc["choices"])
        )
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)
