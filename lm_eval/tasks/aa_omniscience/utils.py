OMNISCIENCE_ANSWER_PROMPT = """\
You are answering questions about {domain}, and in particular {topic}.
You will be given a question, answer with JUST the answer (no explanation).
If you do not know the answer, or you need more context or tools to answer the question,
be clear about this - it is better that you say this than get the wrong answer."""


def _build_prompt(domain: str, topic: str, question: str) -> str:
    instruction = OMNISCIENCE_ANSWER_PROMPT.format(domain=domain, topic=topic)
    return f"{instruction}\n\nQuestion: {question.strip()}"


def _add_fields(doc):
    question = str(doc["question"])
    doc["input"] = _build_prompt(str(doc["domain"]), str(doc["topic"]), question)
    doc["question"] = question
    doc["gold_answer"] = str(doc["answer"])
    return doc


def process_docs(dataset):
    return dataset.map(_add_fields)


def placeholder_metric(references, predictions) -> dict:
    return {"acc": 0.0}
