"""Response collection only.

The task just collects model answers; harmfulness is graded out-of-band by the judge.
The grading data (rubrics, risk_area -> type map, gov/opp thresholds) and the scoring live
in judge_prompts.py / judge_vllm.py

Data: Regional_data_original.jsonl -- "question", "risk_area", "risk_type" (0-7), "id".
"""


def process_docs(dataset):
    return dataset.map(lambda doc: {"risk_type": str(doc.get("risk_type", ""))})


def response_length_metric(predictions, references, **kwargs):
    """Placeholder metric (char length).
    generate_until task just needs a declared metric."""
    pred = predictions[0] if predictions else ""
    return {"response_chars": float(len(pred))}
