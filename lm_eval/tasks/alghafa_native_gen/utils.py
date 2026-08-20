"""Generative variant of AlGhafa-Native (OALL/AlGhafa-Arabic-LLM-Benchmark-Native).

The native benchmark is log-likelihood multiple-choice. Instruct/think checkpoints are scored
generatively in this pipeline, so the model is asked to emit the option letter inside double
square brackets and the letter is extracted by a filter, mirroring the other *_gen tasks.

The Arabic instruction and the question/option rendering are kept identical to the official
OALL prompt; only the answer format differs (Arabic letters + [[X]] instead of a numbered
continuation), because a generated answer has to be parseable.

Choice counts differ per subset (2, 3, 4 or 5 sol* columns), so the letter set is built per
document instead of being hardcoded.
"""

import re

BRACKETED_LABEL_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")

# Arabic option letters, in the order OALL uses them
ARABIC_LETTERS = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي"]

# the official AlGhafa-Native instruction, verbatim
INSTRUCTION = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"


def _choice_keys(doc):
    """Return the sol* keys in numeric order (sol1, sol2, ... )."""
    keys = [k for k in doc.keys()
            if re.fullmatch(r"sol\d+", str(k)) and doc[k] is not None]
    return sorted(keys, key=lambda k: int(k[3:]))


def _choices(doc):
    return [str(doc[k]).strip() for k in _choice_keys(doc)]


def _letters(doc):
    n = len(_choices(doc))
    if n < 2 or n > len(ARABIC_LETTERS):
        raise ValueError(f"AlGhafa doc has {n} choices, expected 2..{len(ARABIC_LETTERS)}")
    return ARABIC_LETTERS[:n]


def _answer_format(letters):
    """The allowed-outputs instruction, matching the other *_gen tasks' phrasing."""
    if len(letters) == 2:
        options = f"[[{letters[0]}]] أو [[{letters[1]}]]"
    else:
        options = "، ".join(f"[[{l}]]" for l in letters[:-1])
        options += f"، أو [[{letters[-1]}]]"
    return (
        "# التعليمات\n"
        "أعد فقط الإجابة النهائية على شكل حرف الخيار داخل أقواس مربعة مزدوجة.\n"
        f"المخرجات المسموح بها فقط هي:\n{options}"
    )


def _options_block(doc):
    letters = _letters(doc)
    return "".join(f"{l}. {c}\n" for l, c in zip(letters, _choices(doc)))


def doc_to_text(doc):
    letters = _letters(doc)
    query = f"{INSTRUCTION}السؤال: {str(doc['query']).strip()}\n"
    query += _options_block(doc)
    query += "الإجابة:"
    return f"{_answer_format(letters)}\n{query}"


def doc_to_target(doc):
    """The Arabic letter of the gold option. `label` is a string index into sol1..solN."""
    letters = _letters(doc)
    idx = int(str(doc["label"]).strip())
    if not 0 <= idx < len(letters):
        raise ValueError(f"AlGhafa label {idx} out of range for {len(letters)} choices")
    return letters[idx]


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
