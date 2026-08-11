import datasets


LETTER_INDICES_AR = ["أ", "ب", "ج", "د"]
INSTRUCTION = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"


def load_alyah_category(**kwargs):
    category = kwargs.get("category")
    if category is None:
        raise ValueError("Alyah category must be provided in task metadata.")

    dataset = datasets.load_dataset("tiiuae/alyah-emirati-benchmark", "default")
    filtered_test = dataset["test"].filter(
        lambda doc: doc.get("category") == category
    )
    return {"test": filtered_test}


def doc_to_text(doc):
    query = f"{INSTRUCTION}السؤال: {doc['query']}\n"

    for index, option_key in enumerate(["option_1", "option_2", "option_3", "option_4"]):
        query += f"{LETTER_INDICES_AR[index]}) {doc[option_key]}\n"
    header = build_prompt(instruction_lang="ar", labels_lang="ar", num_choices=4)
    return header + "\n" + query + "الإجابة:"


def doc_to_choice(doc):
    return LETTER_INDICES_AR


def doc_to_target(doc):
    return LETTER_INDICES_AR[int(doc["correct_answer"]) - 1]


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


#! make sure to add process_docs: !function utils.add_options_with_text and make sure there is no conflict in all subtasks.
def add_options_with_text(dataset):
    def _add_field(doc):
        #! implement the logic to convert options to text
        options_text = "\n".join([f"{LETTER_INDICES_AR[i]}) {doc[f'option_{i+1}']}" for i in range(4)])
        doc["options_with_text"] = options_text
        return doc
    return dataset.map(_add_field)
