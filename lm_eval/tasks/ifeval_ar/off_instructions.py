# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Library of instructions."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

import logging  # absl -> stdlib logging (avoid absl dependency)
import langdetect
import unicodedata

from lm_eval.tasks.ifeval_ar import off_instructions_util as instructions_util
from collections import Counter


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

# The relational operation for comparison.
_COMPARISON_RELATION = ("less than", "at least")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe.", 
    ".إجابتي هي نعم.", "إجابتي هي لا.", "إجابتي هي ربما"
)


_STARTER_OPTIONS = (
    # English Starters
    "I would say", "My answer is", "I believe", 
    "In my opinion", "I think", "I reckon", "I feel", 
    "From my perspective", "As I see it", "According to me", 
    "As far as I'm concerned", "To my understanding", 
    "In my view", "My take on it is", "As per my perception", 
    
    # Arabic Counterparts
    "أود أن أقول", "إجابتي هي", "أعتقد", 
    "في رأيي", "أظن", "أتصور", "أشعر", 
    "من وجهة نظري", "كما أراه", "حسب اعتقادي", 
    "بقدر ما يهمني", "وفقًا لفهمي", 
    "من منظوري", "تقييمي هو", "حسب تصوري"
)

# The options of ending keywords.
_ENDING_OPTIONS = (
    # English Endings
    "Any other questions?", 
    "Is there anything else I can help with?",

    # Arabic Counterparts
    "هل هناك أي أسئلة أخرى؟", 
    "هل هناك أي شيء آخر يمكنني مساعدتك به؟",

    # Additional Arabic Phrases
    "هل لديك استفسار آخر؟", 
    "هل هناك ما ترغب في معرفته أيضًا؟", 
    "هل يمكنني تقديم مساعدة إضافية؟", 
    "هل هناك شيء آخر تحتاج إليه؟", 
    "هل هناك أمر آخر أستطيع مساعدتك به؟", 
    "هل تحتاج إلى توضيح آخر؟", 
    "هل لديك أي طلب آخر؟"
)

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("Section", "SECTION")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500

_FREQUENCY = 10

class Instruction:
  """An instruction template."""

  def __init__(self, instruction_id):
    self.id = instruction_id

  def build_description(self, **kwargs):
    raise NotImplementedError("`build_description` not implemented.")

  def get_instruction_args(self):
    raise NotImplementedError("`get_instruction_args` not implemented.")

  def get_instruction_args_keys(self):
    raise NotImplementedError("`get_instruction_args_keys` not implemented.")

  def check_following(self, value):
    raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
  """Check the language of the entire response."""

  def build_description(self, *, language = None):
    """Build the instruction description.

    Args:
      language: A string representing the expected language of the response. The
        language has to comply to the 97 types defined in
        `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
        ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
        for example, `en` for English, `zh` for Chinese, `fr` for French.

    Returns:
      A string representing the instruction description.
    """
    self._language = language
    if self._language is None:
      self._language = random.choice(list(_LANGUAGES.keys()))
    self._description_pattern = (
        "Your ENTIRE response should be in {language} language, no other " +
        "language is allowed.")
    return self._description_pattern.format(language=_LANGUAGES[self._language])

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"language": self._language}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["language"]

  def check_following(self, value):
    """Check if the language of the entire response follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the language of `value` follows instruction; otherwise False.
    """
    assert isinstance(value, str)

    try:
      return langdetect.detect(value) == self._language
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True


class NumberOfSentences(Instruction):
  """Check the number of sentences."""

  def build_description(self, *, num_sentences = None,
                        relation = None):
    """Build the instruction description.

    Args:
      num_sentences: An integer specifying the number of sentences as a
        threshold.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of sentences < the threshold;
        if 'at least', the actual number of sentences >= the threshold.

    Returns:
      A string representing the instruction description.
    """
    # The number of sentences as a threshold for comparison.
    self._num_sentences_threshold = num_sentences
    if (self._num_sentences_threshold is None or
        self._num_sentences_threshold < 0):
      self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Your response should contain {relation} {num_sentences} sentences.")
    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_sentences=self._num_sentences_threshold)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_sentences", "relation"]

  def check_following(self, value):
    """Check if the number of sentences follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the response follows the instruction.

    Raise:
        ValueError if the string in `instruction_args` is not in
        [`less_than`, `at_least`].
    """
    num_sentences = instructions_util.count_sentences(value)
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_sentences < self._num_sentences_threshold
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
  """Check the placeholders in template writing."""

  def build_description(self, *, num_placeholders = None):
    """Build the instruction description.

    Args:
      num_placeholders: An integer denoting the minimum number of
        placeholders required in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
    self._description_pattern = (
        "The response must contain at least {num_placeholders} placeholders " +
        "represented by square brackets, such as [address].")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_placeholders": self._num_placeholders}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_placeholders"]

  def check_following(self, value):
    """Check if the number of placeholders follows the instruction.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual number of placeholders in the response is greater than
      or equal to `num_placeholders`; otherwise, False.
    """
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
  """Checks the bullet list in the prompt."""

  def build_description(self, *, num_bullets = None):
    """Build the instruction description.

    Args:
      num_bullets: An integer specifying the exact number of bullet lists
        that is required to appear in the response.

    Returns:
      A string representing the instruction description.
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "Your answer must contain exactly {num_bullets} bullet points. " +
        "Use the markdown bullet points such as:\n" +
        "* This is point 1. \n" +
        "* This is point 2")
    return self._description_pattern.format(
        num_bullets=self._num_bullets)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_bullets": self._num_bullets}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_bullets"]

  def check_following(self, value):
    r"""Check if the number of bullet lists meets the requirement.

    Args:
      value: A string representing the response. The response is expected to
        contain some bullet lists that start with `\*`.

    Returns:
      True if the actual number of bullet lists in the response meets the
      requirement.
    """
    bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
    return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
  """Checks the constrained response.""" 

  def build_description(self):
    """Build the instruction description."""
    # A sequence of string(s) representing the options of the expected response.
    self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
    self._description_pattern = (
        "Answer with one of the following options: {response_options}")
    return self._description_pattern.format(
        response_options=self._constrained_responses)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response matches the constrained options.

    Args:
      value: A string representing the response.

    Returns:
      True if the actual response contains one of the options in the constrained
      responses; otherwise False.
    """
    value = value.strip()
    for constrained_response in self._constrained_responses:
      if constrained_response in value:
        return True
    return False


class ConstrainedStartChecker(Instruction):
  """Checks the response start."""

  def build_description(self, *, starter = None):
    """Build the instruction description.

    Args:
      starter: A string representing the keyward that the response should start
        with.

    Returns:
      A string representing the instruction description.
    """
    self._starter = starter.strip() if isinstance(starter, str) else starter
    if self._starter is None:
      self._starter = random.choice(_STARTER_OPTIONS)
    self._description_pattern = (
        "During the conversation, when it is your turn, " +
        "please always start with {starter}")
    return self._description_pattern.format(starter=self._starter)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"starter": self._starter}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["starter"]

  def check_following(self, value):
    """Checks if the response starts with the constrained keyword or phrase.

    Args:
      value: A string representing the response.

    Returns:
      True if the response starts with the given phrase or keyword that is
      contained in `instruction_args`; otherwise, False.
    """
    response_pattern = r"^\s*" + self._starter + r".*$"
    response_with_constrained_start = re.search(response_pattern, value,
                                                flags=re.MULTILINE)
    return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
  """Checks the highlighted section."""

  def build_description(self, *, num_highlights = None):
    """Build the instruction description.

    Args:
      num_highlights: An integer specifying the minimum number of highlighted
        sections.

    Returns:
      A string representing the instruction description.
    """
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    self._description_pattern = (
        "Highlight at least {num_highlights} sections in your answer with " +
        "markdown, i.e. *highlighted section*.")

    return self._description_pattern.format(num_highlights=self._num_highlights)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_highlights": self._num_highlights}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_highlights"]

  def check_following(self, value):
    """Checks if the number of highlighted sections meets the requirement.

    Args:
      value: a string repesenting the response. The response is expected to
        contain highlighted sections in the format of *highlighted*.

    Returns:
      True if the actual number of highlighted sections in the format of
      *highlighed sections* meets the minimum requirement; otherwise False.
    """
    num_highlights = 0
    highlights = re.findall(r"\*[^\n\*]*\*", value)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
    for highlight in highlights:
      if highlight.strip("*").strip():
        num_highlights += 1
    for highlight in double_highlights:
      if highlight.removeprefix("**").removesuffix("**").strip():
        num_highlights += 1

    return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
  """Checks the sections."""

  def build_description(self, *, section_spliter = None,
                        num_sections = None):
    """Build the instruction description.

    Args:
      section_spliter: A string represents the section spliter keyword that
        marks a new section, i.e., `Section` or `SECTION`.
      num_sections: An integer specifying the number of sections.

    Returns:
      A string representing the instruction description.
    """
    self._section_spliter = section_spliter.strip() if isinstance(
        section_spliter, str) else section_spliter
    if self._section_spliter is None:
      self._section_spliter = random.choice(_SECTION_SPLITER)

    self._num_sections = num_sections
    if self._num_sections is None or self._num_sections < 0:
      self._num_sections = random.randint(1, _NUM_SECTIONS)

    self._description_pattern = (
        "Your response must have {num_sections} sections. Mark the beginning " +
        "of each section with {section_spliter} X, such as:\n" +
        "{section_spliter} 1\n" +
        "[content of section 1]\n" +
        "{section_spliter} 2\n" +
        "[content of section 2]")

    return self._description_pattern.format(
        num_sections=self._num_sections,
        section_spliter=self._section_spliter)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"section_spliter": self._section_spliter,
            "num_sections": self._num_sections}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["section_spliter", "num_sections"]

  def check_following(self, value):
    """Checks the response contains multiple sections.

    Args:
      value: A string representing the response. The response is expected
        to contain multiple sections (number of sections is greater than 1).
        A new section starts with `Section 1`, where the number denotes the
        section index.

    Returns:
      True if the number of sections in the response is greater than or equal to
      the minimum number of sections; otherwise, False.
    """
    section_splitter_patten = r"\s?" + self._section_spliter  + r"\s?\d+\s?"
    sections = re.split(section_splitter_patten, value)
    num_sections = len(sections) - 1
    return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
  """Checks the paragraphs."""

  def build_description(self, *, num_paragraphs = None):
    """Build the instruction description.

    Args:
      num_paragraphs: An integer specifying the number of paragraphs.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._description_pattern = (
        "There should be {num_paragraphs} paragraphs. " +
        "Paragraphs are separated with the markdown divider: ***")

    return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_paragraphs": self._num_paragraphs}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_paragraphs"]

  def check_following(self, value):
    """Checks the response contains required number of paragraphs.

    Args:
      value: A string representing the response. The response may contain
        paragraphs that are separated by the markdown divider: `***`.

    Returns:
      True if the actual number of paragraphs is the same as required;
      otherwise, False.
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num_paragraphs = len(paragraphs)

    for index, paragraph in enumerate(paragraphs):
      if not paragraph.strip():
        if index == 0 or index == len(paragraphs) - 1:
          num_paragraphs -= 1
        else:
          return False

    return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
  """Checks the postscript."""

  def build_description(self, *, postscript_marker = None
                        ):
    """Build the instruction description.

    Args:
      postscript_marker: A string containing the keyword that marks the start
        of the postscript section.

    Returns:
      A string representing the instruction description.
    """
    self._postscript_marker = postscript_marker.strip() if isinstance(
        postscript_marker, str) else postscript_marker
    if self._postscript_marker is None:
      self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

    self._description_pattern = (
        "At the end of your response, please explicitly add a postscript " +
        "starting with {postscript}")

    return self._description_pattern.format(postscript=self._postscript_marker)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"postscript_marker": self._postscript_marker}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["postscript_marker"]

  def check_following(self, value):
    """Checks if the response follows the postscript format.

    Args:
      value: a string representing the response. The response is expected to
        contain a postscript section.

    Returns:
      True if the response contains a postscript section starting with
      the keyword containing in the `instruction_args`; otherwise False.
    """
    value = value.lower()
    if self._postscript_marker == "P.P.S":
      postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif self._postscript_marker == "P.S.":
      postscript_pattern = r"\s*p\.\s?s\..*$"
    else:
      postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
    postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
    return True if postscript else False


#########################################################################################################################################################
class TashkeelChecker(Instruction):
    """Check the presence of specific Tashkeel marks by name with a specified count."""

    # Mapping of Tashkeel names to their Unicode characters
    TASHKEEL_MAP = {
        "Fathatan": "\u064B",
        "Dammatan": "\u064C",
        "Kasratan": "\u064D",
        "Fatha": "\u064E",
        "Damma": "\u064F",
        "Kasra": "\u0650",
        "Shadda": "\u0651",
        "Sukun": "\u0652",
    }

    def build_description(self, *, tashkeel_name=None, count=1):
        """Build the instruction description.

        Args:
            tashkeel_name: A string representing the name of the Tashkeel mark to check.
            count: An integer indicating the expected frequency of the Tashkeel mark.

        Returns:
            A string representing the instruction description.
        """
        if not tashkeel_name:
            raise ValueError("The 'tashkeel_name' argument must be provided.")

        if tashkeel_name not in self.TASHKEEL_MAP:
            raise ValueError(
                f"Invalid Tashkeel name. Choose from: {', '.join(self.TASHKEEL_MAP.keys())}."
            )

        if not isinstance(count, int) or count < 0:
            raise ValueError("The 'count' argument must be a non-negative integer.")

        self._tashkeel_name = tashkeel_name
        self._tashkeel = self.TASHKEEL_MAP[tashkeel_name]
        self._count = count

        self._description_pattern = (
            "Check for the presence of Tashkeel '{tashkeel_name}' in the text with a frequency of {count}."
        )

        return self._description_pattern.format(
            tashkeel_name=self._tashkeel_name, count=self._count
        )

    def get_instruction_args(self):
        """Returns the keyword args of build_description."""
        return {"tashkeel_name": self._tashkeel_name, "count": self._count}

    def get_instruction_args_keys(self):
        """Returns the args keys of build_description."""
        return ["tashkeel_name", "count"]

    def check_following(self, value):
        """Check if the response contains the specified Tashkeel by name and count.

        Args:
            value: The response string to check.

        Returns:
            A boolean indicating if the response satisfies the condition.
        """
        if not isinstance(value, str):
            raise ValueError("The 'value' argument must be a string.")

        # Normalize the text to ensure consistent Unicode representation
        normalized_text = unicodedata.normalize("NFKD", value)
        
        # Count combining characters (diacritics) using Unicode properties
        char_counts = Counter(c for c in normalized_text if unicodedata.combining(c))
        
        # Get the count of the specific tashkeel mark
        tashkeel_count = char_counts.get(self._tashkeel, 0)

        return tashkeel_count == self._count

#########################################################################################################################################################
class KeywordListChecker(Instruction):
    """Check the existence of certain keywords with specified mode."""

    def build_description(self, *, keywords=None, mode="all"):
        """Build the instruction description.

        Args:
            keywords: A list of strings representing the keywords that are
                expected in the response.
            mode: A string indicating the matching condition. Options are:
                "any" - At least one keyword must exist.
                "all" - All keywords must exist.
                "none" - None of the keywords must exist.

        Returns:
            A string representing the instruction description.
        """
        if not keywords:
            self._keywords = instructions_util.generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            if not isinstance(keywords, list):
                raise ValueError("The 'keywords' argument must be a list of strings.")
            self._keywords = keywords

        self._keywords = sorted(self._keywords)
        self._mode = mode.lower()
        if self._mode not in ["any", "all", "none"]:
            raise ValueError("Invalid mode. Choose from 'any', 'all', or 'none'.")

        self._description_pattern = (
            "Include keywords {keywords} in the response under mode '{mode}'."
        )

        return self._description_pattern.format(
            keywords=self._keywords, mode=self._mode
        )

    def get_instruction_args(self):
        """Returns the keyword args of build_description."""
        return {"keywords": self._keywords, "mode": self._mode}

    def get_instruction_args_keys(self):
        """Returns the args keys of build_description."""
        return ["keywords", "mode"]

    def check_following(self, value):
        """Check if the response contains the expected keywords based on the mode.

        Args:
            value: The response string to check.

        Returns:
            A boolean indicating if the response satisfies the condition.
        """
        # Helper function to normalize Arabic text: remove tashkeel and strip "ال"
        def normalize_arabic(text):
            tashkeel_unicode = [
                "\u064B",  # Fathatan
                "\u064C",  # Dammatan
                "\u064D",  # Kasratan
                "\u064E",  # Fatha
                "\u064F",  # Damma
                "\u0650",  # Kasra
                "\u0651",  # Shadda
                "\u0652",  # Sukun
            ]
            text = re.sub(r"ال", "", text)  # Remove "ال" prefix
            for tashkeel in tashkeel_unicode:
                text = text.replace(tashkeel, "")  # Remove tashkeel
            return text

        # Normalize both the keywords and the response
        normalized_value = normalize_arabic(value)
        normalized_keywords = [normalize_arabic(keyword) for keyword in self._keywords]

        found_keywords = [
            keyword for keyword in normalized_keywords if re.search(keyword, normalized_value, flags=re.IGNORECASE)
        ]

        if self._mode == "any":
            return bool(found_keywords)
        elif self._mode == "all":
            return len(found_keywords) == len(normalized_keywords)
        elif self._mode == "none":
            return not bool(found_keywords)

class KeywordChecker(Instruction):
  """Check the exisitence of certain keywords."""

  def build_description(self, *, keywords = None
                        ):
    """Build the instruction description.

    Args:
      keywords: A sequence of strings representing the keywords that are
        expected in the response.

    Returns:
      A string representing the instruction description.
    """

    if not keywords:
      self._keywords = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = ("Include keywords {keywords} in the response.")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keywords"]

  def check_following(self, value):
    """Check if the response contain the expected keywords."""
    for keyword in self._keywords:
      if not re.search(keyword, value, flags=re.IGNORECASE):
        return False
    return True


class KeywordFrequencyChecker(Instruction):
  """Check the keyword frequency."""

  def build_description(self, *, keyword = None,
                        frequency = None,
                        relation = None):
    """Build the instruction description.

    Args:
      keyword: A string representing a keyword that is expected in the response.
      frequency: An integer specifying the number of times `keyword` is expected
        to appear in the response.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of occurrences < frequency;
        if 'at least', the actual number of occurrences >= frequency.

    Returns:
      A string representing the instruction description.
    """
    if not keyword:
      self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
    else:
      self._keyword = keyword.strip()

    self._frequency = frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "In your response, the word {keyword} should appear {relation} " +
        "{frequency} times.")

    return self._description_pattern.format(
        keyword=self._keyword,
        relation=self._comparison_relation,
        frequency=self._frequency)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["keyword", "frequency", "relation"]

  def check_following(self, value):
    """Checks if the response contain the keyword with required frequency."""
    actual_occurrences = len(re.findall(
        self._keyword, value, flags=re.IGNORECASE))

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return actual_occurrences < self._frequency
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
  """Checks the number of words."""

  def build_description(self, *, num_words = None,
                        relation = None):
    """Build the instruction description.

    Args:
      num_words: An integer specifying the number of words contained in the
        response.
      relation: A string in (`less than`, `at least`), defining the relational
        operator for comparison.
        Two relational comparisons are supported for now:
        if 'less than', the actual number of words < num_words;
        if 'at least', the actual number of words >= num_words.

    Returns:
      A string representing the instruction description.
    """

    self._num_words = num_words
    if self._num_words is None or self._num_words < 0:
      self._num_words = random.randint(
          _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
      )

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("The supported relation for comparison must be in "
                       f"{_COMPARISON_RELATION}, but {relation} is given.")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "Answer with {relation} {num_words} words.")

    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_words=self._num_words)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_words": self._num_words,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_words", "relation"]

  def check_following(self, value):
    """Checks if the response contains the expected number of words."""
    num_words = instructions_util.count_words(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_words < self._num_words
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_words >= self._num_words


class JsonFormat(Instruction):
  """Check the Json format."""

  def build_description(self):
    self._description_pattern = (
        "Entire output should be wrapped in JSON format. You can use markdown"
        " ticks such as ```."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as _:
      return False
    return True


class ParagraphFirstWordCheck(Instruction):
  """Check the paragraph and the first word of the nth paragraph."""

  def build_description(self, num_paragraphs = None,
                        nth_paragraph = None,
                        first_word = None):
    r"""Build the instruction description.

    Args:
      num_paragraphs: An integer indicating the number of paragraphs expected
        in the response. A paragraph is a subset of the string that is
        expected to be separated by '\n\n'.
      nth_paragraph: An integer indicating the paragraph number that we look at.
        Note that n starts from 1.
      first_word: A string that represent the first word of the bth paragraph.

    Returns:
      A string representing the instruction description.
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._nth_paragraph = nth_paragraph
    if (
        self._nth_paragraph is None
        or self._nth_paragraph <= 0
        or self._nth_paragraph > self._num_paragraphs
    ):
      self._nth_paragraph = random.randint(1, self._num_paragraphs)

    self._first_word = first_word
    if self._first_word is None:
      self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]
    self._first_word = self._first_word.lower()

    self._description_pattern = (
        "There should be {num_paragraphs} paragraphs. " +
        "Paragraphs and only paragraphs are separated with each other by two " +
        "new lines as if it was '\\n\\n' in python. " +
        "Paragraph {nth_paragraph} must start with word {first_word}.")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs,
        nth_paragraph=self._nth_paragraph,
        first_word=self._first_word)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["num_paragraphs", "nth_paragraph", "first_word"]

  def check_following(self, value):
    """Checks for required number of paragraphs and correct first word.

    Args:
      value: a string representing the response. The response may contain
        paragraphs that are separated by two new lines and the first word of
        the nth paragraph will have to match a specified word.

    Returns:
      True if the number of paragraphs is the same as required and the first
      word of the specified paragraph is the same as required. Otherwise, false.
    """

    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
      if not paragraph.strip():
        num_paragraphs -= 1

    # check that index doesn't go out of bounds
    if self._nth_paragraph <= num_paragraphs:
      paragraph = paragraphs[self._nth_paragraph - 1].strip()
      if not paragraph:
        return False
    else:
      return False

    first_word = ""
    punctuation = {".", ",", "?", "!", "'", '"', "،", "؛", "؟", "…" }


    # get first word and remove punctuation
    word = paragraph.split()[0].strip()
    word = word.lstrip("'")
    word = word.lstrip('"')

    for letter in word:
      if letter in punctuation:
        break
      first_word += letter.lower()

    return (
        num_paragraphs == self._num_paragraphs
        and first_word == self._first_word
    )




class ForbiddenWords(Instruction):
  """Checks that specified words are not used in response."""

  def build_description(self, forbidden_words = None
                        ):
    """Build the instruction description.

    Args:
      forbidden_words: A sequences of strings respresenting words that are not
        allowed in the response.

    Returns:
      A string representing the instruction description.
    """

    if not forbidden_words:
      self._forbidden_words = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._forbidden_words = list(set(forbidden_words))
    self._forbidden_words = sorted(self._forbidden_words)
    self._description_pattern = (
        "Do not include keywords {forbidden_words} in the response."
    )

    return self._description_pattern.format(
        forbidden_words=self._forbidden_words
    )

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"forbidden_words": self._forbidden_words}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["forbidden_words"]

  def check_following(self, value):
    """Check if the response does not contain the expected keywords."""
    for word in self._forbidden_words:
      if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
        return False
    return True


class RephraseParagraph(Instruction):
  """Checks that the paragraph is rephrased."""

  def build_description(self, *, original_paragraph, low, high
                        ):
    """Builds the instruction description.

    Args:
      original_paragraph: A string presenting the original paragraph. The
        rephrases response should have betweeb low-high words in common.
      low: An integer presenting the lower bound of similar words.
      high: An integer representing the upper bound of similar words.

    Returns:
      A string representing the instruction description.
    """
    self._original_paragraph = original_paragraph
    self._low = low
    self._high = high

    self._description = ("Rephrase the following paragraph: " +
                         "{original_paragraph}\nYour response should have " +
                         "between {low} and {high} of the same words. " +
                         "Words are the same if and only if all of the " +
                         "letters, ignoring cases, are the same. For " +
                         "example, 'run' is the same as 'Run' but different " +
                         "to 'ran'.")

    return self._description.format(original_paragraph=original_paragraph,
                                    low=self._low, high=self._high)

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return {"original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["original_paragraph", "low", "high"]

  def check_following(self, value):
    val_words = re.findall(r"\w+", value.lower())
    original_words = re.findall(r"\w+", self._original_paragraph.lower())
    similar_words = 0

    dict_val = collections.Counter(val_words)
    dict_original = collections.Counter(original_words)

    for word in dict_original:
      similar_words += min(dict_original[word], dict_val[word])

    return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
  """Check that two responses were given."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Give two different responses. Responses and only responses should"
        " be separated by 6 asterisk symbols: ******."
    )
    return self._description_pattern

  def get_instruction_args(self):
    """Returns the keyward args of `build_description`."""
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response has two different answers.

    Args:
      value: A string representing the response.

    Returns:
      True if two responses are detected and false otherwise.
    """
    valid_responses = list()
    responses = value.split("******")
    for index, response in enumerate(responses):
      if not response.strip():
        if index != 0 and index != len(responses) - 1:
          return False
      else:
        valid_responses.append(response)
    return (
        len(valid_responses) == 2
        and valid_responses[0].strip() != valid_responses[1].strip()
    )


class RepeatPromptThenAnswer(Instruction):
  """Checks that Prompt is first repeated then answered."""

  def build_description(self, *, prompt_to_repeat = None):
    """Build the instruction description.

    Args:
      prompt_to_repeat: The prompt that is meant to be repeated.

    Returns:
      A string representing the instruction description.
    """
    if not prompt_to_repeat:
      raise ValueError("prompt_to_repeat must be set.")
    else:
      self._prompt_to_repeat = prompt_to_repeat
    self._description_pattern = (
        "First repeat the request word for word without change,"
        " then give your answer (1. do not say any words or characters"
        " before repeating the request; 2. the request you need to repeat"
        " does not include this sentence)"
    )
    return self._description_pattern

  def get_instruction_args(self):
    return {"prompt_to_repeat": self._prompt_to_repeat}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["prompt_to_repeat"]

  def check_following(self, value):
    if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
      return True
    return False


class EndChecker(Instruction):
  """Checks that the prompt ends with a given phrase."""

  def build_description(self, *, end_phrase = None):
    """Build the instruction description.

    Args:
      end_phrase: A string representing the phrase the response should end with.

    Returns:
      A string representing the instruction description.
    """
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
    )
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "Finish your response with this exact phrase {ender}. "
        "No other words should follow this phrase.")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    return {"end_phrase": self._end_phrase}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["end_phrase"]

  def check_following(self, value):
    """Checks if the response ends with the expected phrase."""
    value = value.strip().strip("\"").lower()
    self._end_phrase = self._end_phrase.strip().lower()
    return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
  """Checks the response for a title."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Your answer must contain a title, wrapped in double angular brackets,"
        " such as <<poem of joy>>."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks if the response contains a title."""
    pattern = r"<<[^\n]+>>|«[^\n]+?»"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("<").rstrip(">").strip():
        return True
    return False


class LetterFrequencyChecker(Instruction):
  """Checks letter frequency."""

  def build_description(self, *, letter = None,
                        let_frequency = None,
                        let_relation = None):
    """Build the instruction description.

    Args:
      letter: A string representing a letter that is expected in the response.
      let_frequency: An integer specifying the number of times `keyword` is
        expected to appear in the response.
      let_relation: A string in (`less than`, `at least`), defining the
        relational operator for comparison. Two relational comparisons are
        supported for now; if 'less than', the actual number of
        occurrences < frequency; if 'at least', the actual number of
        occurrences >= frequency.

    Returns:
      A string representing the instruction description.
    """
    if (
        not letter
        or len(letter) > 1
        or ord(letter.lower()) < 97
        or ord(letter.lower()) > 122
    ):
      self._letter = random.choice(list(string.ascii_letters))
    else:
      self._letter = letter.strip()
    self._letter = self._letter.lower()

    self._frequency = let_frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _LETTER_FREQUENCY)

    if let_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif let_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {let_relation} is given."
      )
    else:
      self._comparison_relation = let_relation

    self._description_pattern = (
        "In your response, the letter {letter} should appear {let_relation}"
        " {let_frequency} times."
    )

    return self._description_pattern.format(
        letter=self._letter,
        let_frequency=self._frequency,
        let_relation=self._comparison_relation,
    )

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {"letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["letter", "let_frequency", "let_relation"]

  def check_following(self, value):
    """Checks that the response contains the letter at the right frequency."""
    value = value.lower()
    letters = collections.Counter(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return letters[self._letter] < self._frequency
    else:
      return letters[self._letter] >= self._frequency


class LetterListFrequencyChecker(Instruction):
    """Checks word frequency based on starting letters and position."""

    def build_description(self, *, letters=None, frequency=None,
                          relation=None, position=None):
        """Build the instruction description.

        Args:
            letters: A list of letters to check if words start with any of them.
            word_frequency: An integer specifying the number of words matching the criteria.
            word_relation: A string in (less than, at least), defining the relational operator.
            position: A string in (start, end, any), specifying where to check the letters in the words.

        Returns:
            A string representing the instruction description.
        """
        if not letters or not all(isinstance(letter, str) and len(letter) == 1 for letter in letters):
            raise ValueError("Letters must be a list of single-character strings.")
        self._letters = [letter.lower() for letter in letters]

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {relation} is given."
            )
        else:
            self._comparison_relation = relation

        if position is None:
            self._position = random.choice(["start", "end", "any"])
        elif position not in ["start", "end", "any"]:
            raise ValueError(
                "Position must be one of 'start', 'end', or 'any', but "
                f"'{position}' is given."
            )
        else:
            self._position = position

        self._description_pattern = (
            "Write a sentence in which {relation} {frequency} words {position} with any of "
            "the letters in the list {letters}."
        )

        return self._description_pattern.format(
            
            relation=self._comparison_relation,
            frequency=self._frequency,
            position=f"start with" if self._position == "start" else
            f"end with" if self._position == "end" else "contain",
            letters=self._letters,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letters": self._letters,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
            "position": self._position,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of build_description."""
        return ["letters", "frequency", "relation", "position"]

    def check_following(self, sentence):
        """Checks that the sentence matches the instruction."""
        tashkeel_unicode = [
            "\u064B",  # Fathatan
            "\u064C",  # Dammatan
            "\u064D",  # Kasratan
            "\u064E",  # Fatha
            "\u064F",  # Damma
            "\u0650",  # Kasra
            "\u0651",  # Shadda
            "\u0652",  # Sukun
        ]

        # Helper function to normalize Arabic text: remove tashkeel and "ال"
        def normalize_arabic(word):
            word = re.sub(r"^ال", "", word)  # Remove "ال" prefix
            for tashkeel in tashkeel_unicode:
                word = word.replace(tashkeel, "")  # Remove tashkeel
            return word

        words = [normalize_arabic(word.lower()) for word in sentence.split() if word.strip()]
        if not words:
            return False
        match_count = 0

        for word in words:
            if not word:
                continue
            if self._position == "start" and len(word) > 0 and word[0] in self._letters:
                match_count += 1
            elif self._position == "end" and len(word) > 0 and word[-1] in self._letters:
                match_count += 1
            elif self._position == "any" and  any(letter in word for letter in self._letters):
                match_count += 1

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return match_count < self._frequency
        else:
            return match_count >= self._frequency

class WordLetterChecker(Instruction):
    """Checks word frequency based on starting or ending letters.""" 
    def __init__(self, instruction_id="word_letter_checker"):
        super().__init__(instruction_id)
        self._letters = []
        self._frequency = 0
        self._comparison_relation = ""
        self._check_type = "starts_with"
    def build_description(self, *, letters=None, let_frequency=None, let_relation=None, check_type="starts_with"):
        """Build the instruction description.

        Args:
          letters: A list of letters to check against. Defaults to random letters if not provided.
          let_frequency: An integer specifying the number of words expected to start or end with the letters.
          let_relation: A string in (`less than`, `at least`), defining the relational operator for comparison.
          check_type: A string (`starts_with`, `ends_with`) specifying whether to check word starts or ends.

        Returns:
          A string representing the instruction description.
        """
        # Validate letters
        if not letters or not all(isinstance(letter, str) and len(letter) == 1 for letter in letters):
            self._letters = random.sample(list(string.ascii_lowercase), 3)
        else:
            self._letters = [letter.lower().strip() for letter in letters]

        # Validate frequency
        self._frequency = let_frequency if let_frequency is not None and let_frequency >= 0 else random.randint(1, 10)

        valid_relations = ["less than", "at least"]
        if let_relation not in valid_relations:
            raise ValueError(f"The supported relation for comparison must be in {valid_relations}, but {let_relation} is given.")
        self._comparison_relation = let_relation

        if check_type not in ["starts_with", "ends_with"]:
            raise ValueError("`check_type` must be either 'starts_with' or 'ends_with'.")
        self._check_type = check_type

        self._description_pattern = (
            "In your response, {let_relation} {let_frequency} words should {check_type} one of the letters: {letters}."
        )

        return self._description_pattern.format(
            let_relation=self._comparison_relation,
            let_frequency=self._frequency,
            check_type="start with" if self._check_type == "starts_with" else "end with",
            letters=", ".join(self._letters)
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letters": self._letters,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
            "check_type": self._check_type
        }

    def check_following(self, value):
        """Checks that the response satisfies the letter conditions.

        Args:
          value: A string containing the response text.

        Returns:
          A boolean indicating if the conditions are satisfied.
        """
        words = [word for word in value.lower().split() if word.strip()]
        if self._check_type == "starts_with":
            count = sum(1 for word in words if len(word) > 0 and word[0] in self._letters)
        elif self._check_type == "ends_with":
            count = sum(1 for word in words if len(word) > 0 and word[-1] in self._letters)
        else:
            count = 0

        if self._comparison_relation == "less than":
            return count < self._frequency
        else:  # "at least"
            return count >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
  """Checks that the response is in english and is in all capital letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Your entire response should be in English, and in all capital letters."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response is in English and in all capital letters."""
    assert isinstance(value, str)

    try:
      return value.isupper() and langdetect.detect(value) == "en"
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True


class LowercaseLettersEnglishChecker(Instruction):
  """Checks that the response is in english and is in all lowercase letters."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "Your entire response should be in English, and in all lowercase"
        " letters. No capital letters are allowed."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response is in English and in all lowercase letters."""
    assert isinstance(value, str)

    try:
      return value.islower() and langdetect.detect(value) == "en"
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "Unable to detect language for text %s due to %s", value, e
      )  # refex: disable=pytotw.037
      return True


class CommaChecker(Instruction):
  """Checks the response for no commas."""

  def build_description(self):
    """Build the instruction description."""
    self._description_pattern = (
        "In your entire response, refrain from the use of any commas."
    )
    return self._description_pattern

  def get_instruction_args(self):
    return None

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return []

  def check_following(self, value):
    """Checks that the response does not contain commas."""
    return not re.search(r"[\,،]", value)


class CapitalWordFrequencyChecker(Instruction):
  """Checks frequency of words with all capital letters."""

  def build_description(
      self,
      capital_frequency = None,
      capital_relation = None,
  ):
    """Build the instruction description.

    Args:
      capital_frequency: An integer that represents the number of words that
        should be in all capital letters.
      capital_relation: A string that is 'at least' or 'at most' that refers to
        the frequency.

    Returns:
      A string representing the instruction description.
    """
    self._frequency = capital_frequency
    if self._frequency is None:
      self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

    self._comparison_relation = capital_relation
    if capital_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif capital_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "The supported relation for comparison must be in "
          f"{_COMPARISON_RELATION}, but {capital_relation} is given."
      )

    self._description_pattern = (
        "In your response, words with all capital letters should appear"
        " {relation} {frequency} times."
    )

    return self._description_pattern.format(
        frequency=self._frequency, relation=self._comparison_relation
    )

  def get_instruction_args(self):
    """Returns the keyword args of build description."""
    return {
        "capital_frequency": self._frequency,
        "capital_relation": self._comparison_relation,
    }

  def get_instruction_args_keys(self):
    """Returns the args keys of `build_description`."""
    return ["capital_frequency", "capital_relation"]

  def check_following(self, value):
    """Checks the frequency of words with all capital letters."""
    words = instructions_util.nltk.word_tokenize(value)
    capital_words = [word for word in words if word.isupper()]

    capital_words = len(capital_words)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return capital_words < self._frequency
    else:
      return capital_words >= self._frequency

class QuotationChecker(Instruction):
    """Checks response is wrapped with various double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            "Wrap your entire response with appropriate double quotation marks, such as ASCII quotes, "
            "curly quotes, or Arabic quotation marks."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with recognized double quotation marks."""
        # Strip leading and trailing whitespace
        value = value.strip()

        # Define accepted quotation mark pairs
        quotation_pairs = [
            ('"', '"'),        # ASCII double quotes
            ('“', '”'),        # Curly double quotes
            ('«', '»'),        # French/Arabic angle quotes
            ('„', '“')         # Curly lower and upper quotes (used in some languages)
        ]

        # Check if value is at least 2 characters long and wrapped by any of the accepted pairs
        return (
            len(value) > 1 and 
            any(value.startswith(open_q) and value.endswith(close_q) for open_q, close_q in quotation_pairs)
        )


class Check_Word_Repetition(Instruction):
    """Check if any word is repeated more than twice consecutively."""

    def build_description(self):
        """Build the instruction description."""
        self._description = (
            "Your response should not contain any word repeated more than "
            "twice consecutively."
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Check if any word is repeated more than twice consecutively.

        Args:
          value: A string representing the response.

        Returns:
          True if the response does not contain repeated words; False otherwise.
        """
        assert isinstance(value, str)
        words = value.split()
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return False
        return True

class Check_Sentence_Repetition(Instruction):
    """Check if any sentence is repeated more than once."""

    def build_description(self):
        """Build the instruction description."""
        self._description = (
            "Your response should not contain any sentence repeated more than "
            "once."
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Check if any sentence is repeated more than once.

        Args:
          value: A string representing the response.

        Returns:
          True if the response does not contain repeated sentences; False otherwise.
        """
        assert isinstance(value, str)
        sentences = instructions_util.split_into_sentences(value)
        sentence_counts = {}
        for sentence in sentences:
            sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
            if sentence_counts[sentence] > 1:
                return False
        return True

class Check_Punctuation(Instruction):
    """Check if the punctuation used in the response matches the specified language."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the language of punctuation to check.
            Must be either 'ar' for Arabic or 'en' for English.

        Returns:
          A string representing the instruction description.
        """
        if language not in ["ar", "en"]:
            raise ValueError("The language must be either 'ar' for Arabic or 'en' for English.")
        
        self._language = language
        self._description_pattern = (
            "Your response should use {language} punctuation marks only."
        )
        lang_desc = "Arabic" if self._language == "ar" else "en"
        return self._description_pattern.format(language=lang_desc)

    def get_instruction_args(self):
        """Returns the keyword args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the punctuation in the response matches the specified language.

        Args:
          value: A string representing the response.

        Returns:
          True if the punctuation matches the language or no punctuation is found; False otherwise.
        """
        assert isinstance(value, str)
        assert self._language in ["ar", "en"], "Invalid language specified."

        arabic_punctuation = set("،؛؟!")
        english_punctuation = set(",;?!.")

        response_punctuation = {char for char in value if char in arabic_punctuation or char in english_punctuation}

        if not response_punctuation:
            return True

        if self._language == "ar":
            return response_punctuation.issubset(arabic_punctuation)
        elif self._language == "en":
            return response_punctuation.issubset(english_punctuation)

