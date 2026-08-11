LEGAL_INSTRUCTION = (
    "أجب عن السؤال القانوني التالي بإيجاز باللغة العربية الفصحى في سياق قانون "
    "دولة الإمارات العربية المتحدة. أجب بالإجابة فقط."
)


def process_docs_qa(dataset):
    def _add(doc):
        doc["question"] = doc["question"]
        doc["input"] = f"{LEGAL_INSTRUCTION}\n\n{doc['question']}"
        return doc

    return dataset.map(_add)


def process_docs_rag(dataset):
    def _add(doc):
        combined = f"{doc['context']}\n\n{doc['question']}"
        doc["question"] = combined
        doc["input"] = f"{LEGAL_INSTRUCTION}\n\n{combined}"
        return doc

    return dataset.map(_add)


def placeholder_metric(references, predictions) -> dict:
    return {"acc": 0.0}
