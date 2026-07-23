"""Arabic IFEval instruction verifiers.

#### Hasan: Arabic IFEval (inceptionai/Arabic_IFEval, 404 prompts, 23 instruction types).
20 of the 23 instruction types are language-agnostic and REUSE the English IFEval classes
verbatim (`lm_eval.tasks.ifeval.instructions`) — see instructions_registry_ar.py. This module
holds only the deltas:

  Arabic OVERRIDES of English classes (same kwargs/description, Arabic-aware check_following):
    * CommaChecker            — also reject the Arabic comma "،" (U+060C).
    * QuotationChecker        — accept Arabic/typographic double-quote wrappers («» " " „ …).
    * NumberOfSentences       — count sentences with the Arabic-aware counter (؟/۔).
    * ConstrainedResponseChecker — Arabic answer options (إجابتي هي نعم./لا./ربما.).

  NEW Arabic-only instructions (no English equivalent):
    * LetterListFrequencyChecker (keywords:letter_list_freq) — N words whose first/last (or any)
      letter is one of a given letter set, compared via relation.
    * TashkeelChecker (detectable_format:tashkeel) — >= N words carrying a given diacritic
      (Fatha/Damma/Kasra/…/tanwin/Shadda).
    * KeywordListExistenceChecker (keywords:list_existence) — response contains all/any of a
      keyword list (diacritic-insensitive).

Faithful REIMPLEMENTATION of the Inception taxonomy (their verifier source is not public);
verifiers are deterministic, so scores are reproducible but not guaranteed identical to the
inceptionai leaderboard (parity caveat, mirrors the ABB gemma-judge caveat).
########
"""

import re

from lm_eval.tasks.ifeval import instructions as en
from lm_eval.tasks.ifeval_ar import instructions_util_ar as ar_util


_COMPARISON_RELATION = en._COMPARISON_RELATION  # ("less than", "at least")


# ═════════════════════════════════════════════════════════════════════════════
# Arabic overrides of English classes
# ═════════════════════════════════════════════════════════════════════════════
class CommaChecker(en.CommaChecker):
    """No commas — Arabic responses use the Arabic comma "،" (U+060C) as well as ",". """

    def check_following(self, value):
        return not re.search(r"[,،]", value)


class QuotationChecker(en.QuotationChecker):
    """Response wrapped in double quotation marks — Arabic text uses several glyphs
    (ASCII ", typographic “ ” „, and the Arabic guillemets « »)."""

    _OPENERS = "\"“„«‹"   # " “ „ « ‹
    _CLOSERS = "\"”»›"          # " ” » ›

    def check_following(self, value):
        value = value.strip()
        return len(value) > 1 and value[0] in self._OPENERS and value[-1] in self._CLOSERS


class NumberOfSentences(en.NumberOfSentences):
    """Sentence count with an Arabic-aware sentence splitter (handles ؟ / ۔)."""

    def check_following(self, value):
        num_sentences = ar_util.count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        return num_sentences >= self._num_sentences_threshold


class ConstrainedResponseChecker(en.ConstrainedResponseChecker):
    """Constrained multiple-choice answer — Arabic option set."""

    #### Hasan: Arabic equivalents of _CONSTRAINED_RESPONSE_OPTIONS (yes / no / maybe).
    _AR_CONSTRAINED_RESPONSE_OPTIONS = (
        "إجابتي هي نعم.",
        "إجابتي هي لا.",
        "إجابتي هي ربما.",
    )
    ########

    def build_description(self):
        self._constrained_responses = self._AR_CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = "أجب بإحدى العبارات التالية: {response_options}"
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )


# ═════════════════════════════════════════════════════════════════════════════
# New Arabic-only instructions
# ═════════════════════════════════════════════════════════════════════════════
class LetterListFrequencyChecker(en.Instruction):
    """keywords:letter_list_freq — count words whose edge letter (start/end) — or any letter,
    when position == 'any' — is in `letters`; compare that count to `frequency` via `relation`."""

    def build_description(self, *, letters=None, frequency=None, relation=None, position=None):
        self._letters = set(letters or [])
        self._frequency = int(frequency) if frequency is not None else 1
        self._position = position if position in ("start", "end", "any") else "end"
        if relation is None:
            self._comparison_relation = _COMPARISON_RELATION[1]
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                f"relation must be in {_COMPARISON_RELATION}, got {relation}"
            )
        else:
            self._comparison_relation = relation
        self._description_pattern = (
            "Include {relation} {frequency} words whose {position} letter is one of {letters}."
        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            frequency=self._frequency,
            position=self._position,
            letters=sorted(self._letters),
        )

    def get_instruction_args(self):
        return {
            "letters": sorted(self._letters),
            "frequency": self._frequency,
            "relation": self._comparison_relation,
            "position": self._position,
        }

    def get_instruction_args_keys(self):
        return ["letters", "frequency", "relation", "position"]

    def check_following(self, value):
        words = ar_util.letter_words(value)
        if self._position == "any":
            count = sum(1 for w in words if any(ch in self._letters for ch in w))
        else:
            count = sum(
                1
                for w in words
                if ar_util.word_edge_letter(w, self._position) in self._letters
            )
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return count < self._frequency
        return count >= self._frequency


class TashkeelChecker(en.Instruction):
    """detectable_format:tashkeel — at least `count` words carry the diacritic `tashkeel_name`
    (Fatha/Damma/Kasra/Fathatan/Dammatan/Kasratan/Shadda/Sukun)."""

    def build_description(self, *, tashkeel_name=None, count=None):
        self._tashkeel_name = tashkeel_name
        self._mark = ar_util.tashkeel_char(tashkeel_name)
        self._count = int(count) if count is not None else 1
        self._description_pattern = (
            "Use at least {count} words containing the diacritic {tashkeel_name}."
        )
        return self._description_pattern.format(
            count=self._count, tashkeel_name=self._tashkeel_name
        )

    def get_instruction_args(self):
        return {"tashkeel_name": self._tashkeel_name, "count": self._count}

    def get_instruction_args_keys(self):
        return ["tashkeel_name", "count"]

    def check_following(self, value):
        if not self._mark:
            return False
        # Count whitespace-delimited tokens that carry the diacritic (do NOT strip diacritics
        # here — we are looking for them).
        num = sum(1 for tok in value.split() if self._mark in tok)
        return num >= self._count


class KeywordListExistenceChecker(en.Instruction):
    """keywords:list_existence — the response must contain all (mode='all') or at least one
    (mode='any') of `keywords`. Matching is diacritic-insensitive on both sides."""

    def build_description(self, *, keywords=None, mode=None):
        self._keywords = list(keywords or [])
        self._mode = mode if mode in ("all", "any") else "all"
        self._description_pattern = "Include {mode} of these keywords: {keywords}."
        return self._description_pattern.format(mode=self._mode, keywords=self._keywords)

    def get_instruction_args(self):
        return {"keywords": self._keywords, "mode": self._mode}

    def get_instruction_args_keys(self):
        return ["keywords", "mode"]

    def check_following(self, value):
        haystack = ar_util.strip_diacritics(value)
        hits = [
            ar_util.strip_diacritics(k) in haystack
            for k in self._keywords
            if k
        ]
        if not hits:
            return True
        return any(hits) if self._mode == "any" else all(hits)
