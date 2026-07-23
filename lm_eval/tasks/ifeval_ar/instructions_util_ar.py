"""Arabic-specific utilities for the Arabic IFEval verifiers.

#### Hasan: new module for the Arabic IFEval integration (inceptionai/Arabic_IFEval).
The English IFEval util (`lm_eval.tasks.ifeval.instructions_util`) is reused where it is
language-agnostic:
  * count_words  -> RegexpTokenizer(r"\\w+") already tokenizes Arabic letter runs fine.
The pieces that are English-specific get an Arabic-aware shim here:
  * count_sentences -> the punkt English tokenizer never breaks on the Arabic question mark
    "؟" (U+061F) or the Urdu full stop "۔" (U+06D4); normalise those to ASCII first.
Plus the primitives the three Arabic-only instructions need (tashkeel/diacritic table,
diacritic stripping, letter-run tokenisation, first/last-letter extraction).
########
"""

import re

from lm_eval.tasks.ifeval import instructions_util as _en_util


# ── sentence / word counting ────────────────────────────────────────────────
# Arabic sentence terminators that the English punkt tokenizer does not know about,
# mapped to their ASCII equivalents so the (reused) English counter breaks on them.
_AR_SENTENCE_TERMINATORS = {
    "؟": "?",   # ؟  Arabic question mark
    "۔": ".",   # ۔  Arabic (Urdu) full stop
    "…": ".",   # …  horizontal ellipsis
}


def count_words(text):
    """Words in an Arabic (or mixed) response. `\\w+` covers Arabic letter runs."""
    return _en_util.count_words(text)


def count_sentences(text):
    """Sentences in an Arabic response — normalise Arabic terminators, then reuse the
    English punkt counter (which handles `.`/`!`/`?`)."""
    for src, dst in _AR_SENTENCE_TERMINATORS.items():
        text = text.replace(src, dst)
    return _en_util.count_sentences(text)


# ── diacritics (tashkeel) ───────────────────────────────────────────────────
# Name -> combining-mark codepoint, covering every `tashkeel_name` in the dataset
# (Kasratan, Dammatan, Fathatan, Shadda, Fatha, Damma, Kasra) plus Sukun for completeness.
# Arabic aliases included so the check is robust to either spelling.
TASHKEEL = {
    "fatha": "َ",       # َ
    "damma": "ُ",       # ُ
    "kasra": "ِ",       # ِ
    "fathatan": "ً",    # ً  tanwin fath
    "dammatan": "ٌ",    # ٌ  tanwin damm
    "kasratan": "ٍ",    # ٍ  tanwin kasr
    "shadda": "ّ",      # ّ
    "sukun": "ْ",       # ْ
    # Arabic-name aliases
    "فتحة": "َ",              # فتحة
    "ضمة": "ُ",                     # ضمة
    "كسرة": "ِ",               # كسرة
    "شدة": "ّ",                     # شدة
    "سكون": "ْ",               # سكون
}

# All Arabic combining diacritics (harakat + tanwin + shadda + sukun + superscript alef).
_DIACRITICS_RE = re.compile(r"[ً-ْٰـ]")  # incl. tatweel U+0640


def tashkeel_char(name):
    """Resolve a `tashkeel_name` (English or Arabic, any case) to its combining char."""
    if not name:
        return None
    return TASHKEEL.get(str(name).strip().lower()) or TASHKEEL.get(str(name).strip())


def strip_diacritics(text):
    """Remove Arabic combining diacritics and the tatweel elongation mark."""
    return _DIACRITICS_RE.sub("", text)


# ── letter-run tokenisation (for keywords:letter_list_freq) ──────────────────
# A "word" for the letter-frequency check is a maximal run of Arabic OR Latin letters.
# Diacritics are stripped first so the first/last *letter* is a base letter.
_LETTER_RUN_RE = re.compile(r"[ء-يٱ-ۓA-Za-z]+")


def letter_words(text):
    """Return the list of letter-run tokens in `text` (diacritics removed)."""
    return _LETTER_RUN_RE.findall(strip_diacritics(text))


def word_edge_letter(word, position):
    """First/last base letter of a token. `position` in {"start","end"}; returns "" if empty."""
    if not word:
        return ""
    return word[0] if position == "start" else word[-1]
