"""Registry for the 23 Arabic IFEval instruction ids.

#### Hasan: maps every instruction_id in inceptionai/Arabic_IFEval to a verifier. 20 reuse the
English IFEval classes unchanged (language-agnostic); 4 use Arabic overrides and 3 are new
Arabic-only checks — all defined in instructions_ar.py. See that module's docstring.
########
"""

from lm_eval.tasks.ifeval import instructions as en
from lm_eval.tasks.ifeval_ar import instructions_ar as ar


INSTRUCTION_DICT = {
    # ── reused verbatim from English IFEval (language-agnostic) ──────────────
    "keywords:existence": en.KeywordChecker,
    "keywords:frequency": en.KeywordFrequencyChecker,
    "keywords:forbidden_words": en.ForbiddenWords,
    "length_constraints:number_words": en.NumberOfWords,
    "length_constraints:number_paragraphs": en.ParagraphChecker,
    "length_constraints:nth_paragraph_first_word": en.ParagraphFirstWordCheck,
    "detectable_content:number_placeholders": en.PlaceholderChecker,
    "detectable_content:postscript": en.PostscriptChecker,
    "detectable_format:number_bullet_lists": en.BulletListChecker,
    "detectable_format:number_highlighted_sections": en.HighlightSectionChecker,
    "detectable_format:multiple_sections": en.SectionChecker,
    "detectable_format:json_format": en.JsonFormat,
    "detectable_format:title": en.TitleChecker,
    "combination:two_responses": en.TwoResponsesChecker,
    "combination:repeat_prompt": en.RepeatPromptThenAnswer,
    "startend:end_checker": en.EndChecker,
    # ── Arabic overrides (language-specific behaviour) ───────────────────────
    "punctuation:no_comma": ar.CommaChecker,
    "startend:quotation": ar.QuotationChecker,
    "length_constraints:number_sentences": ar.NumberOfSentences,
    "detectable_format:constrained_response": ar.ConstrainedResponseChecker,
    # ── new Arabic-only instructions ─────────────────────────────────────────
    "keywords:letter_list_freq": ar.LetterListFrequencyChecker,
    "detectable_format:tashkeel": ar.TashkeelChecker,
    "keywords:list_existence": ar.KeywordListExistenceChecker,
}
