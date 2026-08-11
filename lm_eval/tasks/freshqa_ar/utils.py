import re
import string

# Automated (non-LLM) baseline for FreshQA-Arabic, reported ALONGSIDE the LLM judge. Same coarse
# cover-match idea as the English task, with the Arabic normalization halluscore uses: strip
# harakat + tatweel, unify alef/ya/ta-marbuta variants, drop ASCII and Arabic punctuation.
# FreshQA rows carry SEVERAL acceptable gold answers, so a hit on ANY of them counts.
#
# Deliberately script-agnostic: 109 of the 600 Arabic gold answers are still Latin (proper nouns,
# numbers), and a model answering in Arabic may well render those either way. This metric cannot
# see that nuance -- the FreshEval judge grades by meaning and is the real score; this is only a
# cheap signal that generation produced something on-topic.

# Harakat (U+064B-U+065F), superscript alef (U+0670) and tatweel (U+0640), by explicit codepoint.
# NOTE: written as escapes on purpose. The literal-character form of this class that appears
# elsewhere in the repo ("[<harakat>-<tatweel>]") is a malformed range that spans U+0617-U+064B,
# i.e. the whole Arabic LETTER block -- it deletes the entire string, so any Arabic gold
# normalizes to "" and cover_match can never fire. See halluscore/utils.py.
_ARABIC_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640]")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "؟،؛")


def _normalize(text) -> str:
    text = str(text).lower()
    text = _ARABIC_DIACRITICS.sub("", text)           # strip harakat + tatweel
    text = text.translate(_PUNCT_TABLE)               # ASCII + Arabic punctuation
    text = re.sub(r"[إأآا]", "ا", text)               # unify alef forms
    text = text.replace("ى", "ي").replace("ة", "ه")  # ya / ta-marbuta variants
    return " ".join(text.split())


def freshqa_ar_automated_metrics(predictions, references, **kwargs):
    """cover_match: any normalized gold answer appears as a whitespace-bounded substring of the
    response. Higher is better. See the module note on why this is a floor, not the score."""
    pred = _normalize(predictions[0] if predictions else "")
    if not pred:
        return {"cover_match": 0.0}

    doc = kwargs.get("doc") or {}
    golds = doc.get("answers") or ([references[0]] if references else [])

    padded_pred = f" {pred} "
    for gold in golds:
        gold_norm = _normalize(gold)
        if gold_norm and f" {gold_norm} " in padded_pred:
            return {"cover_match": 1.0}
    return {"cover_match": 0.0}
