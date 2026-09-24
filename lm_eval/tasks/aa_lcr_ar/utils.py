"""AA-LCR Arabic: same prompt construction as the English ``aa_lcr`` task, but the
questions, answers and documents come from the Arabic translation in DATA_DIR
instead of the ArtificialAnalysis/AA-LCR HF repo."""

import json
import os
import unicodedata

import datasets



DATA_DIR = "/jais3/data/jais_2/benchmarks/aa_lcr_ar_data"
QA_FILE = os.path.join(DATA_DIR, "aa_lcr_ar_100_qa_only.json")
DOCS_DIR = os.path.join(DATA_DIR, "documents")

_DOC_TEMPLATE = "BEGIN DOCUMENT {i}:\n{doc}\nEND DOCUMENT {i}"
_PROMPT_TEMPLATE = (
    "BEGIN INPUT DOCUMENTS\n\n"
    "{documents_text}\n\n"
    "END INPUT DOCUMENTS\n\n"
    "أجب عن السؤال التالي باستخدام الوثائق المدخلة المقدمة أعلاه.\n\n"
    "START QUESTION\n\n"
    "{question}\n\n"
    "END QUESTION\n"
)


def _norm_key(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


_DOC_INDEX = None


def _doc_index() -> dict:
    """Map NFC-normalised filename -> Arabic document text (filenames are unique
    across all 30 document sets, so no set_id is needed in the key)."""
    global _DOC_INDEX
    if _DOC_INDEX is None:
        index = {}
        for name in os.listdir(DOCS_DIR):
            path = os.path.join(DOCS_DIR, name)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as f:
                index[_norm_key(name)] = f.read()
        _DOC_INDEX = index
    return _DOC_INDEX


def _load_documents(set_id: str, filenames) -> list[str]:
    """Return the document texts in the given filename order."""
    if isinstance(filenames, str):
        filenames = filenames.split(";")
    index = _doc_index()
    docs = []
    for name in filenames:
        name = _norm_key(name)
        if not name:
            continue
        if name not in index:
            raise KeyError(
                f"AA-LCR-ar document not found in {DOCS_DIR}: set={set_id!r} file={name!r}"
            )
        docs.append(index[name])
    return docs


def _build_prompt(docs: list[str], question: str) -> str:
    documents_text = "\n\n".join(
        _DOC_TEMPLATE.format(i=i + 1, doc=doc) for i, doc in enumerate(docs)
    )
    return _PROMPT_TEMPLATE.format(documents_text=documents_text, question=question.strip())


def load_dataset(**kwargs) -> datasets.DatasetDict:
    """``custom_dataset`` hook: load the 100 Arabic QA rows from the bundled JSON."""
    with open(QA_FILE, encoding="utf-8") as f:
        rows = json.load(f)
    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def _add_fields(doc):
    question = str(doc["question_ar"])
    docs = _load_documents(doc["document_set_id"], doc["data_source_filenames"])
    doc["input"] = _build_prompt(docs, question)
    doc["question"] = question
    doc["gold_answer"] = str(doc["answer_ar"])
    return doc


def process_docs(dataset):
    """``process_docs`` hook: build the prompt for every question (all 100)."""
    return dataset.map(_add_fields)


def placeholder_metric(references, predictions) -> dict:
    return {"acc": 0.0}
