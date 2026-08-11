"""Utilities for the Arabic Linguistic multiple-choice tasks."""

import json
import re
from pathlib import Path
import datasets


DEFAULT_CHOICE_LETTERS = ("A", "B", "C", "D", "E")
LABELED_OPTION_RE = re.compile(r"^\s*([A-E])\s*[.)]\s*")


def load_category(data_files, category_field, category_value, **kwargs):
    dataset = datasets.load_dataset("json", data_files=data_files)
    filtered_test = dataset["test"].filter(
        lambda doc: doc.get(category_field) == category_value
    )
    return {"test": filtered_test}
    
# def load_category(data_files, category_field, category_value, **kwargs):
#     filenames = data_files["test"]
#     if isinstance(filenames, (str, Path)):
#         filenames = [filenames]

#     documents = []

#     for filename in filenames:
#         with Path(filename).open(encoding="utf-8") as file:
#             rows = json.load(file)

#         documents.extend(
#             row
#             for row in rows
#             if row.get(category_field) == category_value
#         )

#     return {"test": datasets.Dataset.from_list(documents)}

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
    header = build_prompt(instruction_lang="ar", labels_lang="en", num_choices=len(letters))
    return header + "\n" + "\n".join(lines)

def doc_to_choice(doc):
    return _choice_letters(doc)


def doc_to_target(doc):
    return doc["answer"]


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
        options = _options(doc)
        letters = _choice_letters(doc)
        lines = []
        for letter, option in zip(letters, options):
            option_text = str(option)
            match = LABELED_OPTION_RE.match(option_text)
            if match and match.group(1) == letter:
                lines.append(option_text)
            else:
                lines.append(f"{letter}. {option_text}")
        
        options_text = "\n".join(lines)
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)

