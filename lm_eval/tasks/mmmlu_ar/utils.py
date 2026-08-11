"""Arabic OpenAI-MMMLU (Omartificial-Intelligence-Space/Arabic_Openai_MMMLU) — generative.

This is OpenAI's MMMLU (the 57-subject MMLU test set, professionally human-translated to Arabic),
the data behind the ILMAAM leaderboard. We evaluate it exactly the way OpenAI does in their
`simple-evals` harness: zero-shot, the English instruction template with the Arabic question and
choices, the model is asked to think then end with 'Answer: $LETTER', and the letter is extracted
with OpenAI's multilingual (Arabic-aware) answer regexes. Suits our thinking checkpoints, which
reason in the trace and emit the final letter in the visible answer.

The prompt template, MULTILINGUAL_ANSWER_* regexes and normalize_extracted_answer are vendored
verbatim from github.com/openai/simple-evals (common.py). Pure `re` only, so this module is safe to
import from any generation venv and from the standalone scorer (mmmlu_ar_eval/mmmlu_score.py).

Leaderboard protocol: the ILMAAM reference run uses 100 questions per subject; process_docs caps to
the first 100 rows of every subject (macro-average over subjects is count-independent, so the cap
does not bias it). The overall in-pipeline metric here is micro exact_match; the leaderboard-headline
macro average (mean over the 57 subjects) and the per-subject breakdown are computed by the scorer.
"""

import re

# ---- OpenAI simple-evals, common.py (verbatim) --------------------------------------------------
QUERY_TEMPLATE_MULTICHOICE = (
    "Answer the following multiple choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think "
    "step by step before answering."
)

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = (
    "(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])"
)
# All the different ways "Answer" is written in different languages
MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Answer\s*:​​​​​​",  # Korean invisible character
    r"উত্তর\s*:",
    r"उत्तर\s*:",
    r"উত্তরঃ",
    r"উত্তর\s*:",
    r"Antwort\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"답\s*:",
    r"答案\s*：",
    r"答案\s*:",
    r"答\s*：",
    r"答\s*:",
    r"答复\s*：",
    r"答曰\s*：",
    r"الإجابة:",
    r"الجواب:",
    r"إجابة:",
    r"الإجابة النهائية:",
    r"الإجابة الصحيحة:",
    r"الإجابة الصحيحة هي:",
    r"الإجابة هي:",
    r"الجواب النهائي:",
    r"Respuesta\s*:",
    r"Risposta\s*:",
    r"答え\s*:",
    r"答え\s*：",
    r"回答\s*:",
    r"回答\s*：",
    r"解答\s*:",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]


def normalize_extracted_answer(extracted_answer: str) -> str:
    return (
        # In arabic these are the letters used for A-D in multiple choice questions
        extracted_answer.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        # In Bengali these are the letters used for A-D in multiple choice questions
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        # In Japanese these are the letters sometimes used for A-D in multiple choice questions
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


# ---- our additions ------------------------------------------------------------------------------
# thinking-trace close tags (configs.THINKING_CLOSING_TAGS) — filtered_resps is already the answer
# for thinking cells, but strip any residual trace defensively before extracting.
_THINK_CLOSE = re.compile(r"</think>|</think_fast>|</think_faster>|<channel\|>|</ifm\|think>")
# fallback: a bare option letter, e.g. "(B)" / "B." / "الإجابة ج" without the "Answer:" prefix
_BARE_LETTER = re.compile(r"(?<![A-Za-z])([A-D]|[أ-د])(?![A-Za-z])")


def strip_thinking(text: str) -> str:
    m = list(_THINK_CLOSE.finditer(text or ""))
    return text[m[-1].end():].lstrip() if m else (text or "")


def extract_letter(response_text: str) -> str:
    """Return the predicted option letter 'A'-'D' (or '' if none), OpenAI simple-evals logic.

    Primary: first MULTILINGUAL_ANSWER_REGEXES match of '<answer-word>: <letter>'. Fallback (ours,
    for free-form thinking output): the last bare option letter in the response. Arabic option
    letters (أ ب ج د) are normalized to A-D."""
    text = strip_thinking(response_text or "")
    if not text.strip():
        return ""

    def _norm(raw):
        # normalize Arabic/CJK option letters to A-D, then upper-case (models may answer 'd')
        letter = normalize_extracted_answer(raw).upper()
        return letter if letter in ("A", "B", "C", "D") else ""

    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, text)
        if match:
            return _norm(match.group(1))
    # fallback: last standalone letter anywhere in the answer
    hits = _BARE_LETTER.findall(text)
    if hits:
        return _norm(hits[-1])
    return ""


# ---- lm-eval task hooks -------------------------------------------------------------------------
def process_docs(dataset):
    """Leaderboard protocol: first 100 questions per subject (subj_idx is the within-subject index
    staged at data-prep time)."""
    return dataset.filter(lambda d: int(d["subj_idx"]) < 100)


def doc_to_text(doc) -> str:
    # f-string interpolation (NOT str.format) so literal braces in the Arabic question/choices are safe
    return (
        f"{QUERY_TEMPLATE_MULTICHOICE}\n\n"
        f"{doc['Question']}\n\n"
        f"A) {doc['A']}\n"
        f"B) {doc['B']}\n"
        f"C) {doc['C']}\n"
        f"D) {doc['D']}"
    )


def doc_to_target(doc) -> str:
    return doc["Answer"]


def extract_answer(resps, docs):
    """lm-eval custom filter: map each raw generation to its extracted letter (or '[invalid]')."""
    def _one(resp_list):
        resp = resp_list[0] if resp_list else ""
        letter = extract_letter(resp if isinstance(resp, str) else "")
        return letter or "[invalid]"
    return map(_one, resps)


def exact_match_letter(predictions, references, **kwargs):
    pred = predictions[0] if predictions else "[invalid]"
    ref = references[0] if references else ""
    return {"exact_match": float(str(pred).strip() == str(ref).strip())}
