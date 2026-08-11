"""Utilities for the ALP Arabic parsing MCQ benchmark."""

from pathlib import Path
import json
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
    
# def load_category(data_filename, category_value, **kwargs):
#     documents = []

#     with Path(data_filename).open(encoding="utf-8") as file:
#         for line in file:
#             if not line.strip():
#                 continue

#             row = json.loads(line)
#             if row.get("category") == category_value:
#                 documents.append(row)

#     return {"test": datasets.Dataset.from_list(documents)}

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
    header = build_prompt(instruction_lang="ar", labels_lang="en", num_choices=len(doc["options"]))
    return header + "\n" + "\n".join(lines)


def doc_to_choice(doc):
    return CHOICE_LETTERS


def doc_to_target(doc):
    answer = doc["answer"]
    if answer not in CHOICE_LETTERS:
        raise ValueError(f"Invalid answer character: {answer!r}")
    return answer


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
        options_text = "\n".join(
            f"{letter}. {option}"
            for letter, option in zip(CHOICE_LETTERS, doc["options"])
        )
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)
