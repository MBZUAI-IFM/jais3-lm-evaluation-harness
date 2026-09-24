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
    valid_options_list = _valid_options(doc)
    header_instruction = build_prompt(instruction_lang="ar", labels_lang="ar", num_choices=len(valid_options_list))
    query = (
        f"{header_instruction}\n"
        f"{INSTRUCTION}\nالسياق:\n{doc.get('Context')}\n"
        f"السؤال:\n{doc['Question']}\n"
    )
    query += "".join(
        f"{arabic_label}. {option}\n"
        for _, arabic_label, option in valid_options_list
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
    if labels_lang not in ["en", "ar"]:
        raise ValueError("labels_lang must be 'en' or 'ar'")
    if num_choices < 2 or num_choices > 26:
        raise ValueError("num_choices must be between 2 and 26")

    # Generate letters
    letters_en = [chr(ord('A') + i) for i in range(num_choices)]
    letters_ar = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي",
                  "ك", "ل", "م", "ن", "س", "ع", "ف", "ص", "ق", "ر",
                  "ش", "ت", "ث", "خ", "ذ", "ض"][:num_choices]

    # Choose label set
    letters = letters_en if labels_lang == "en" else letters_ar

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
        valid_options_list = _valid_options(doc)
        options_text = "".join(
        f"{arabic_label}. {option}\n"
        for _, arabic_label, option in valid_options_list
    )
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)
