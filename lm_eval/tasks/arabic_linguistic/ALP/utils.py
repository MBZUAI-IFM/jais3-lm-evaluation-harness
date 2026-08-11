"""Utilities for the ALP Arabic parsing MCQ benchmark."""

from pathlib import Path

import datasets

CHOICE_LETTERS = ["A", "B", "C", "D"]

def load_category(data_filename, category_value, **kwargs):
    dataset = datasets.load_dataset(
        "json",
        data_files={"test": str(data_filename)},
    )
    return {
        "test": dataset["test"].filter(
            lambda doc: doc.get("category") == category_value
        )
    }


def doc_to_text(doc):
    lines = [
        (
            "أجب عن السؤال باستخدام الخيار المناسب من بين A أو B أو C أو D. "
            "الرجاء الرد بالحرف الصحيح فقط دون أي شرح أو معلومات إضافية."
        ),
        f"السؤال: {doc['question']}",
        f"الجملة: {doc['sentence']}",
        "الاختيارات:",
    ]
    lines.extend(
        f"{letter}. {option}"
        for letter, option in zip(CHOICE_LETTERS, doc["options"])
    )
    lines.append("الإجابة:")
    return "\n".join(lines)


def doc_to_choice(doc):
    return CHOICE_LETTERS


def doc_to_target(doc):
    answer = doc["answer"]
    if answer not in CHOICE_LETTERS:
        raise ValueError(f"Invalid answer character: {answer!r}")
    return answer
