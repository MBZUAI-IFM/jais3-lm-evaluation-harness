"""Language + repetition GATE from inceptionai-abudhabi/IFEval evaluation.py — VERBATIM.

The official `test_language_check` logic that produces the leaderboard's `strict_with_lang`
metric: a prompt's instructions are all zeroed if the response is not in the required language
(>=95% Arabic for lang==["ar"]; mixed allowed for ["ar","en"]) OR contains repetition (>=7
repeated words / >=3 repeated sentences / repeated 5-60-grams — anti-padding). Uses the `regex`
module (\\p{L}); applied offline at score time (rescore_official.py) in the .eval venv, not inside
the lm-eval task (keeps the generation venvs free of the regex/absl deps).
"""
import re
import regex

from lm_eval.tasks.ifeval_ar.off_clean import clean_for_english


def is_valid_arabic_text(text, allowed_english_percentage=0.05):
    arabic_pattern = (r'^[؀-ۿݐ-ݿ'
                      r'ࢠ-ࣿﭐ-﷿ﹰ-﻿]+$')
    english_pattern = r'^[a-zA-Z]+$'
    seqs = regex.findall(r'\p{L}+', text)
    if not seqs:
        return False
    total = len(seqs)
    ar = en = 0
    for s in seqs:
        if regex.fullmatch(arabic_pattern, s):
            ar += 1
        elif regex.fullmatch(english_pattern, s):
            en += 1
        else:
            return False
    if en / total > allowed_english_percentage:
        return False
    return ar > 0


def is_valid_english_text(text):
    if any(c in "،؛؟" for c in text):
        return False
    if re.search(r'[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-ﻯ]', text):
        return False
    for pat in [r'[֐-׿]', r'[Ѐ-ӿ]', r'[一-鿿]',
                r'[぀-ゟ]', r'[゠-ヿ]', r'[가-힯]']:
        if re.search(pat, text):
            return False
    return True


def is_valid_ar_en_text(text):
    arabic_blocks = (r'؀-ۿݐ-ݿࢠ-ࣿ'
                     r'ﭐ-﷿ﹰ-﻿₀-₉')
    allowed = r'[A-Za-z' + arabic_blocks + r']+'
    if not regex.search(allowed, text):
        return False
    for s in regex.findall(r'\p{L}+', text):
        if not regex.fullmatch(allowed, s):
            return False
    return True


def is_correct_language(response, lang):
    ll = sorted(set(lang))
    if not ll:
        return True
    if set(ll) == {"ar", "en"}:
        return is_valid_ar_en_text(response)
    single = ll[0]
    if single == "ar":
        return is_valid_arabic_text(response)
    if single == "en":
        return is_valid_english_text(response)
    raise ValueError(f"Unsupported languages: {lang}")


def has_repeated_sentences(response, max_repeats=2):
    sentences = [s.strip() for s in re.split(r'[.!?]', response) if s.strip()]
    for i in range(len(sentences) - max_repeats):
        if len(sentences[i].split()) < 3:
            continue
        if all(sentences[i] == sentences[i + j] for j in range(max_repeats + 1)):
            return True
    return False


def has_repeated_words(response, min_repeats=7):
    words = re.compile(r'\b[0-9A-Za-zء-ي]+\b').findall(response)
    count = 1
    for prev, curr in zip(words, words[1:]):
        if curr == prev:
            count += 1
            if count >= min_repeats:
                return True
        else:
            count = 1
    return False


def has_consecutive_repeated_ngrams(response, min_n=5):
    text = re.sub(r'[^\w\s]', '', response)
    words = re.findall(r'\b\w+\b', text)
    total = len(words)
    if total < 3 * min_n:
        return False
    max_n = min(total // 3, 60)
    for n in range(min_n, max_n + 1):
        for i in range(total - 3 * n + 1):
            base = tuple(words[i:i + n])
            count = 1
            j = i + n
            while j + n <= total and tuple(words[j:j + n]) == base:
                count += 1
                j += n
            if count >= 3:
                return True
    return False


def passes_language_check(raw, lang):
    """Official test_language_check verdict for one response."""
    response = clean_for_english(raw)
    if "P.P.S." in response:
        response = response.split("P.P.S.")[0]
    response = re.sub(r'^```(?:\w*)\n?', '', response)
    response = re.sub(r'\n?```$', '', response)
    response = re.sub(r'[{}\[\]:,"]', '', response)
    response = re.sub(r'[₀₁₂₃₄₅₆₇₈₉`*]', '', response)
    ok = is_correct_language(response, lang or ["en"])
    if ok and (has_repeated_words(response) or has_repeated_sentences(response)
               or has_consecutive_repeated_ngrams(response)):
        ok = False
    return ok
