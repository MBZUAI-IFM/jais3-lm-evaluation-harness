import re
import string

# Automated (non-LLM) baseline for FreshQA, reported ALONGSIDE the LLM judge. FreshQA rows carry
# SEVERAL acceptable gold answers (up to 10 — aliases, alternate phrasings, "no such thing"
# rebuttals for false-premise rows), so cover_match credits a hit on ANY of them. The real score
# is the FreshEval-strict ternary grade written by judge_vllm.py under separate keys, which is
# what survives the merge; this only gives a cheap sanity signal that generation worked.

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text) -> str:
    text = str(text).lower()
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def freshqa_automated_metrics(predictions, references, **kwargs):
    """cover_match: any normalized gold answer appears as a whitespace-bounded substring of the
    response. A coarse proxy only — it cannot see staleness, cannot tell an abstention from a
    confabulation, and credits a gold string buried in a wrong answer. Higher is better."""
    pred = _normalize(predictions[0] if predictions else "")
    if not pred:
        return {"cover_match": 0.0}

    # `references` is doc_to_target (the first gold answer). The full alias list lives on the doc,
    # so use it when lm-eval passes the doc through, else fall back to the single reference.
    doc = kwargs.get("doc") or {}
    golds = doc.get("answers") or ([references[0]] if references else [])

    padded_pred = f" {pred} "
    for gold in golds:
        gold_norm = _normalize(gold)
        if gold_norm and f" {gold_norm} " in padded_pred:
            return {"cover_match": 1.0}
    return {"cover_match": 0.0}
