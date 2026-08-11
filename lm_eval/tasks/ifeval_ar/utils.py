"""Arabic IFEval in-task scorer — official verifiers (inceptionai-abudhabi/IFEval).

Uses Inception's exact verifier registry (off_instructions_registry, vendored verbatim) and their
response normalisation (off_clean.clean_for_english), so the numbers are methodologically identical
to the official leaderboard. Two jais-specific touches: strip the model's thinking trace to the
final answer before checking (reasoning_effort checkpoints; no-op for non-thinking models), and
decode `kwargs` (a JSON string in the staged data).

Emits the strict prompt- & instruction-level accuracy WITHOUT the language/repetition gate. The
official leaderboard metric is `strict_with_lang` = this, gated by off_gate.passes_language_check
(zeroes a prompt whose response isn't valid Arabic or repeats). That gate uses `regex`, so — like
ABB's out-of-pipeline scoring — it is applied after generation by
hasan.iqbal/ifeval_ar_eval/rescore_official.py (run in the .eval venv), keeping the generation
venvs free of extra deps. Run the rescore for leaderboard-comparable numbers.
"""
import json
import re

from lm_eval.tasks.ifeval_ar import off_instructions_registry as instructions_registry
from lm_eval.tasks.ifeval_ar.off_clean import clean_for_english


_THINKING_CLOSING_TAGS = re.compile(
    r"</think>|</think_fast>|</think_faster>|<channel\|>|</ifm\|think>"
)


def _strip_thinking(text):
    if not text:
        return text
    m = list(_THINKING_CLOSING_TAGS.finditer(text))
    return text[m[-1].end():].lstrip() if m else text


def _decode_kwargs(raw):
    return json.loads(raw) if isinstance(raw, str) else raw


def _follow_list(doc, cleaned_response):
    kwargs = _decode_kwargs(doc["kwargs"])
    following = []
    for idx, instruction_id in enumerate(doc["instruction_id_list"]):
        inst = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        inst.build_description(**(kwargs[idx] or {}))
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=doc["prompt"])
        following.append(bool(cleaned_response.strip() and inst.check_following(cleaned_response)))
    return following


def process_results(doc, results):
    answer = _strip_thinking(results[0])
    cleaned = clean_for_english(answer)
    following = _follow_list(doc, cleaned)
    return {
        "prompt_level_strict_acc": all(following) and len(following) > 0,
        "inst_level_strict_acc": following,
    }


def agg_inst_level_acc(items):
    flat = [x for sub in items for x in sub]
    return sum(flat) / len(flat)
