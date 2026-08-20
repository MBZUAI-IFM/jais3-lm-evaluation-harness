"""Placeholder in-task metric for ABB. ABB's real score is a bespoke per-question mix of
manual rules + LLM-as-judge (the official evaluate_and_score), reproduced in the standalone
scorer hasan.iqbal/abb_eval/abb_score.py which runs on the generated samples. lm-eval requires
a metric on a generate_until task, so this just reports the non-empty-generation rate as a
sanity signal; it is NOT the ABB score."""


def abb_generated_baseline(predictions, references, **kwargs):
    prediction = predictions[0] if predictions else ""
    return {"generated": float(bool((prediction or "").strip()))}
