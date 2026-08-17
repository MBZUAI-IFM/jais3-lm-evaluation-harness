"""Arabic EXAMS (OALL/Arabic_EXAMS), generative variant for instruct/think checkpoints.

Same subject-aware instruction and question/option rendering as the official OALL prompt; the
only change is the answer format, because a generated answer has to be parseable: the model is
asked for the option letter inside double square brackets and a filter extracts it. This mirrors
the other *_gen tasks (3LM_gen, MedArabiQ_gen, alghafa_native_gen).

Log-likelihood scoring is degenerate for chat/thinking checkpoints (measured on Arabic MMMLU:
0-shot LL put gemma-2-9b-it at chance), which is why the pair exists.
"""

import re

BRACKETED_LABEL_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")

# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي"]
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# fmt: on

CHOICE_KEYS = ["A", "B", "C", "D"]


def _letters():
    return LETTER_INDICES_AR[: len(CHOICE_KEYS)]


def _answer_format():
    letters = _letters()
    options = "، ".join(f"[[{l}]]" for l in letters[:-1]) + f"، أو [[{letters[-1]}]]"
    return (
        "# التعليمات\n"
        "أعد فقط الإجابة النهائية على شكل حرف الخيار داخل أقواس مربعة مزدوجة.\n"
        f"المخرجات المسموح بها فقط هي:\n{options}"
    )


def _options_block(doc):
    letters = _letters()
    return "".join(f" {l}) {doc[k]}\n" for l, k in zip(letters, CHOICE_KEYS))


def doc_to_text(doc):
    instruction = (
        "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول "
        f"{str(doc['subject']).replace('_', ' ')}. \n\n"
    )
    query = f"{instruction}السؤال: {str(doc['question']).strip()}\n"
    query += _options_block(doc)
    query += "\nالإجابة:"
    return f"{_answer_format()}\n{query}"


def doc_to_target(doc):
    """The Arabic letter of the gold option. `answer` is a Latin letter A-D."""
    idx = LETTER_INDICES.index(str(doc["answer"]).strip().upper())
    return _letters()[idx]


def add_options_with_text(dataset):
    """`options_with_text` is what judge_registry's mcq_prompt reads when the task is judged."""
    def _add(doc):
        doc["options_with_text"] = _options_block(doc)
        return doc
    return dataset.map(_add)


def extract_bracketed_label(resps, docs):
    def extract(resp):
        if not isinstance(resp, str):
            return "[invalid]"
        matches = BRACKETED_LABEL_RE.findall(resp)
        if not matches:
            return "[invalid]"
        return matches[-1]

    return map(lambda r: extract(r[0] if r else ""), resps)


def _normalize(label):
    # strip the tatweel form so هـ and ه compare equal
    return str(label).strip().replace("هـ", "ه").lower()


def exact_match_normalized_label(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else "[invalid]"
    reference = references[0] if references else "[invalid]"
    return {"exact_match": float(_normalize(prediction) == _normalize(reference))}
