"""Utilities for the Arabic Linguistic multiple-choice tasks."""

import re

import datasets


DEFAULT_CHOICE_LETTERS = ("A", "B", "C", "D", "E")
LABELED_OPTION_RE = re.compile(r"^\s*([A-E])\s*[.)]\s*")


def load_category(data_files, category_field, category_value, **kwargs):
    dataset = datasets.load_dataset("json", data_files=data_files)
    filtered_test = dataset["test"].filter(
        lambda doc: doc.get(category_field) == category_value
    )
    return {"test": filtered_test}


def _options(doc):
    options = doc.get("options", doc.get("choices"))
    if not isinstance(options, list) or not options:
        raise ValueError("Document must contain a non-empty 'options' or 'choices' list.")
    return options


def _choice_letters(doc):
    options = _options(doc)
    letters = doc.get("choice_letters")
    if letters is None:
        letters = list(DEFAULT_CHOICE_LETTERS[: len(options)])

    if len(letters) != len(options):
        raise ValueError(
            f"Choice label count ({len(letters)}) does not match option count "
            f"({len(options)})."
        )
    return letters


def doc_to_text(doc):
    options = _options(doc)
    letters = _choice_letters(doc)
    allowed_letters = " أو ".join(letters)
    lines = [
        (
            "أجب عن السؤال باستخدام الخيار المناسب من بين "
            f"{allowed_letters}. الرجاء الرد بالحرف الصحيح فقط دون أي شرح "
            "أو معلومات إضافية."
        )
    ]

    sentence = doc.get("sentence")
    if sentence:
        lines.append(f"السياق: {sentence}")

    lines.extend((f"السؤال: {doc['question']}", "الاختيارات:"))
    for letter, option in zip(letters, options):
        option_text = str(option)
        match = LABELED_OPTION_RE.match(option_text)
        if match and match.group(1) == letter:
            lines.append(option_text)
        else:
            lines.append(f"{letter}. {option_text}")

    lines.append("الإجابة:")
    return "\n".join(lines)


def doc_to_choice(doc):
    return _choice_letters(doc)


def doc_to_target(doc):
    target = doc.get("answer_idx", doc.get("answer"))
    if isinstance(target, bool) or not isinstance(target, int):
        raise ValueError("Document answer must be an integer index.")

    option_count = len(_options(doc))
    if target < 0 or target >= option_count:
        raise ValueError(
            f"Answer index {target} is outside the range of {option_count} options."
        )
    return target
