"""In-task coverage metric for the ABB subtasks.

ABB's real 0-10 score is produced after generation by judge_vllm: `judge_registry.prompts.abb_prompt`
grades each question's LLM-as-judge rubric and `judge_registry.scores.abb_scorer` blends that with
ABB's manual scoring rules. The manual rules deliberately do NOT run here: an in-task metric only
receives (doc, results) and cannot reach `generation_metadata`, so on a thinking checkpoint -- where
the harness leaves `resps` empty whenever the model emits no closing think tag -- every manual rule
would score 0 while the judged rubrics still saw the recovered answer. The scorer reads the response
through `get_model_response`, which has that fallback, so both halves see the same text.

What is left here is a coverage signal only. It replaces the old `generated` metric, which was the
only number ABB ever published: with no judge_registry entry for the task, the dashboard showed this
non-empty-generation rate as if it were the score, so every model that answered all 470 questions
read as ~100 against an ABB leaderboard that tops out at 9.25/10.
"""


def abb_non_empty_rate(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else ""
    return {"non_empty_rate": float(bool((prediction or "").strip()))}
