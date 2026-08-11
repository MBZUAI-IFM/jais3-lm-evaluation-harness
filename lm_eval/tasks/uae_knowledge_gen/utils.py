import re

_ALPHA = ["A", "B", "C", "D", "E"]

# "A.  text" / "A) text" / "(A) . text" -> letter + text.
_OPTION_PREFIX_RE = re.compile(r"^\s*\(?\s*([A-Za-z])\s*\)?\s*[.):\-]\s*")
# Leading letter of a `correct_answer`, incl. the bare-letter form ("B").
_GOLD_LETTER_RE = re.compile(r"^\s*\(?\s*([A-Za-z])\s*\)?\s*\.?")
_WS_RE = re.compile(r"\s+")
BRACKETED_LABEL_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")


def _squash(text) -> str:
    """Collapse the double spaces / stray newlines the source file carries."""
    return _WS_RE.sub(" ", str(text)).strip()


def _split_option(option):
    """('A', 'text') for 'A.  text' and '(A) . text'; (None, text) if unprefixed."""
    match = _OPTION_PREFIX_RE.match(str(option))
    if not match:
        return None, _squash(option)
    return match.group(1).upper(), _squash(str(option)[match.end():])


def _gold_letter(correct_answer, letters, texts):
    match = _GOLD_LETTER_RE.match(str(correct_answer))
    if match:
        letter = match.group(1).upper()
        if letter in letters:
            return letter
    _, gold_text = _split_option(correct_answer)
    if gold_text in texts:
        return letters[texts.index(gold_text)]
    raise ValueError(
        f"uae_knowledge: cannot resolve gold letter for correct_answer={correct_answer!r} "
        f"against option letters {letters}"
    )


def process_docs(dataset):
    def _prep(doc):
        letters, texts = [], []
        for i, option in enumerate(doc["options"]):
            letter, text = _split_option(option)
            letters.append(letter or _ALPHA[i])
            texts.append(text)
        options = [f"{letter}. {text}" for letter, text in zip(letters, texts)]
        return {
            "question": _squash(doc["question"]),
            "options": options,                      
            "options_with_text": "\n".join(options),   
            "answer": _gold_letter(doc["correct_answer"], letters, texts),
        }

    return dataset.map(_prep)


def build_prompt(num_choices: int) -> str:
    letters = _ALPHA[:num_choices] if num_choices <= len(_ALPHA) else [
        chr(ord("A") + i) for i in range(num_choices)
    ]
    options = "، ".join(f"[[{letter}]]" for letter in letters[:-1])
    options += f"، أو [[{letters[-1]}]]"
    return (
        "# التعليمات\n"
        "أعد فقط الإجابة النهائية على شكل حرف الخيار داخل أقواس مربعة مزدوجة.\n"
        f"المخرجات المسموح بها فقط هي:\n{options}"
    )


def doc_to_text(doc):
    return (
        f"{build_prompt(len(doc['options']))}\n"
        "أجب عن السؤال التالي باختيار الإجابة الصحيحة من بين الخيارات المتاحة.\n"
        f"السؤال: {doc['question']}\n"
        "الاختيارات:\n"
        f"{doc['options_with_text']}\n"
        "الإجابة:"
    )


def doc_to_target(doc):
    return doc["answer"]


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
        "exact_match": float(normalize_label(prediction) == normalize_label(reference))
    }
