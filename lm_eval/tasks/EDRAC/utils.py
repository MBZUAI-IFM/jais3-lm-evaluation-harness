"""Utilities for the EDRAC lm-evaluation-harness tasks."""

from __future__ import annotations

from collections.abc import Iterable

import datasets
from rouge_score import rouge_scorer, tokenizers


DATASET_PATH = "MBZUAI/EDRAC"
MAX_QUESTIONS = 10

# Task names follow the paper/repository labels, while the Hub uses `sau` for
# Saudi and `ema` for Emirati.
DIALECT_TO_DATASET_CODE = {
    "egy": "egy",
    "mor": "mor",
    "ksa": "sau",
    "syr": "syr",
    "uae": "ema",
}


class WhitespaceTokenizer(tokenizers.Tokenizer):
    """ROUGE tokenizer used by the official EDRAC evaluation recipe."""

    def tokenize(self, text: str) -> list[str]:
        return str(text).split()


def _is_blank(value: object) -> bool:
    return value is None or not str(value).strip()


def expand_qa_pairs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Expand passage rows into one independent record per usable QA pair."""

    records = []
    for row in dataset:
        passage_id = row["ID"]
        for qa_index in range(1, MAX_QUESTIONS + 1):
            question = row[f"Question {qa_index}"]
            answer = row[f"Answer {qa_index}"]
            question_blank = _is_blank(question)
            answer_blank = _is_blank(answer)

            if question_blank != answer_blank:
                raise ValueError(
                    f"EDRAC row {passage_id!r} has an unmatched question/answer "
                    f"at index {qa_index}."
                )
            if question_blank:
                continue

            records.append(
                {
                    "id": f"{passage_id}__q{qa_index}",
                    "passage_id": passage_id,
                    "dialect": row["Dialect"],
                    "url": row["URL"],
                    "qa_index": qa_index,
                    "passage": row["Passage"],
                    "question": str(question).strip(),
                    "answer": str(answer).strip(),
                }
            )

    return datasets.Dataset.from_list(records)


def load_edrac_dialect(**kwargs) -> dict[str, datasets.Dataset]:
    """Load, filter, and expand one EDRAC dialect selected by task metadata."""

    dialect = kwargs.get("dialect")
    if dialect not in DIALECT_TO_DATASET_CODE:
        supported = ", ".join(DIALECT_TO_DATASET_CODE)
        raise ValueError(
            f"EDRAC dialect must be one of {supported}; received {dialect!r}."
        )

    dataset_code = DIALECT_TO_DATASET_CODE[dialect]
    dataset = datasets.load_dataset(DATASET_PATH)
    filtered_test = dataset["test"].filter(
        lambda doc: doc["Dialect"] == dataset_code
    )
    return {"test": expand_qa_pairs(filtered_test)}


def doc_to_text(doc: dict) -> str:
    """Build the official EDRAC user message for one question."""

    return (
        f"Passage: {doc['passage']}\n\n"
        f"Question: {doc['question']}\n\n"
        "Answer:"
    )


def doc_to_target(doc: dict) -> str:
    return doc["answer"]


def rouge_l(predictions, references, **kwargs) -> tuple[str, str]:
    """Collect one prediction/reference pair for batched ROUGE-L scoring."""

    del kwargs
    prediction = str(predictions[0]) if predictions else ""
    reference = str(references[0]) if references else ""
    return prediction, reference


def rouge_l_aggregate(items: Iterable[tuple[str, str]]) -> float:
    """Return mean whitespace-tokenized ROUGE-L F1 on a 0--100 scale."""

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
        tokenizer=WhitespaceTokenizer(),
    )
    scores = []
    for prediction, reference in items:
        if _is_blank(prediction):
            scores.append(0.0)
            continue
        score = scorer.score(str(reference), str(prediction))["rougeL"].fmeasure
        scores.append(100.0 * score)

    return sum(scores) / len(scores) if scores else 0.0

