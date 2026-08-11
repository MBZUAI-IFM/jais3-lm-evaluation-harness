# AA-Omniscience

Judging needs 3 edits to the shared files.

## 1. `configs.py`
```python
EN_TASKS["aa_omniscience"] = ["instruct", "think"]
BENCHMARK_TO_JUDGE["aa_omniscience"] = "google/gemma-4-26B-A4B-it"  
```

## 2. `judge_prompts.py` — add the ternary grader
```python
def aa_omniscience_prompt(data: dict) -> str:
    doc = data["doc"]
    question = doc["question"]
    gold_answer = data["target"]
    model_response = _clean_response(data["resps"][0][0])
    if not model_response:
        model_response = _clean_response(
            data["generation_metadata"][0][0].get("reasoning", "")
        )[-2000:]
    return f"""\
You are grading a model's answer to a short-answer knowledge question against a gold target answer.
Grade the predicted answer as one of: CORRECT, INCORRECT, or NOT_ATTEMPTED.

The gold target is authoritative; do not re-solve the question. Focus only on whether the model's
FINAL answer matches the gold target.

CORRECT (grade "A"): the predicted answer contains the gold target without contradicting it
(semantic equivalents, minor formatting/spelling, extra non-contradicting detail all count).
INCORRECT (grade "B"): contradicts the gold target or commits to a different/wrong answer.
NOT_ATTEMPTED (grade "C"): refusals, "I don't know", "need more context/tools", empty/unrelated,
or hedging that never commits. The answer prompt invites this, so grade it C, never B.

Output exactly one JSON object, no preamble or fences:
{{
    "reasoning": "<1-3 sentences>",
    "grade": "<one of A, B, C>"
}}

<question>
{question}
</question>

<gold_answer>
{gold_answer}
</gold_answer>

<model_response>
{model_response}
</model_response>"""
```
Register it before the `_gen` matcher:
```python
REGISTRY = ((aa_omniscience_prompt, lambda t: "aa_omniscience" in t),) + REGISTRY

def is_ternary_task(task: str) -> bool:
    return "aa_omniscience" in task or "simpleqa" in task or "halluscore" in task
```

## 3. `judge_vllm.py` — in `simpleqa_scores()` add inside the return dic
```python
"non_hallucination,none": cga,
"omniscience_index,none": (n_correct - n_incorrect) / total if total else 0.0,
```