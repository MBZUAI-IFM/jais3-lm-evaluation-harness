"""Arabic IFEval process_results — strict/loose prompt- and instruction-level accuracy.

#### Hasan: mirrors the English IFEval `utils.py` (lm_eval.tasks.ifeval.utils), with two
Arabic-IFEval-specific changes:
  1. `kwargs` in inceptionai/Arabic_IFEval is a JSON-encoded STRING (a list of per-instruction
     dicts), not a native list — decode it before use.
  2. Strip the model's thinking trace ("<think>…</think>ANSWER") to the final answer before
     checking, so long reasoning never inflates/blows the word/sentence/format verifiers. The
     jais3 vLLM path already strips for "think" runs; this is an idempotent safety net that also
     covers the reasoning_effort=low ("instruct") checkpoints that think silently.
Uses the Arabic registry (instructions_registry_ar) for all 23 instruction ids.
########
"""

import dataclasses
import json
import re
from typing import Dict, Optional, Union

from lm_eval.tasks.ifeval_ar import instructions_registry_ar as instructions_registry


# Closing tags for every thinking format we support (mirrors configs.THINKING_CLOSING_TAGS).
_THINKING_CLOSING_TAGS = re.compile(r"</think>|</think_fast>|</think_faster>|<channel\|>|</ifm\|think>")


def _strip_thinking(text):
    """Return the answer after the LAST thinking-closing tag (or the text unchanged)."""
    if not text:
        return text
    matches = list(_THINKING_CLOSING_TAGS.finditer(text))
    if matches:
        return text[matches[-1].end():].lstrip()
    return text


def _decode_kwargs(raw):
    """Arabic IFEval stores kwargs as a JSON string; accept a native list too."""
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list
    prompt: str
    kwargs: list


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list


def test_instruction_following_strict(inp, response):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None/empty values so build_description doesn't get unexpected kwargs.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp, response):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=_decode_kwargs(doc["kwargs"]),
    )
    response = _strip_thinking(results[0])

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    return sum(flat_items) / len(flat_items)
