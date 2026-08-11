import os
import unicodedata
import zipfile

from huggingface_hub import hf_hub_download


REPO = "ArtificialAnalysis/AA-LCR"
ZIP_FILENAME = "extracted_text/AA-LCR_extracted-text.zip"

_DOC_TEMPLATE = "BEGIN DOCUMENT {i}:\n{doc}\nEND DOCUMENT {i}"
_PROMPT_TEMPLATE = (
    "BEGIN INPUT DOCUMENTS\n\n"
    "{documents_text}\n\n"
    "END INPUT DOCUMENTS\n\n"
    "Answer the following question using the input documents provided above.\n\n"
    "START QUESTION\n\n"
    "{question}\n\n"
    "END QUESTION\n"
)


def _norm_key(s: str) -> str:
    """Recover the real UTF-8 filename (zipfile decodes unflagged names as cp437)
    and NFC-normalize so combining vs precomposed chars match the CSV."""
    try:
        s = s.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return unicodedata.normalize("NFC", s).strip()


_DOC_INDEX = None


def _doc_index() -> dict:
    global _DOC_INDEX
    if _DOC_INDEX is None:
        zip_path = hf_hub_download(REPO, ZIP_FILENAME, repo_type="dataset")
        index = {}
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                parts = info.filename.split("/")
                if len(parts) < 4:
                    continue
                set_id = _norm_key(parts[-2])
                base = _norm_key(os.path.basename(info.filename))
                index[(set_id, base)] = z.read(info).decode("utf-8", errors="replace")
        _DOC_INDEX = index
    return _DOC_INDEX


def _load_documents(set_id: str, filenames) -> list[str]:
    """Return the document texts for ``set_id`` in the given filename order."""
    if isinstance(filenames, str):
        filenames = filenames.split(";")
    index = _doc_index()
    set_id = _norm_key(set_id)
    docs = []
    for name in filenames:
        name = _norm_key(name)
        if not name:
            continue
        key = (set_id, name)
        if key not in index:
            raise KeyError(f"AA-LCR document not found in zip: set={set_id!r} file={name!r}")
        docs.append(index[key])
    return docs


def _build_prompt(docs: list[str], question: str) -> str:
    documents_text = "\n\n".join(
        _DOC_TEMPLATE.format(i=i + 1, doc=doc) for i, doc in enumerate(docs)
    )
    return _PROMPT_TEMPLATE.format(documents_text=documents_text, question=question.strip())


def _add_fields(doc):
    question = str(doc["question"])
    docs = _load_documents(doc["document_set_id"], doc["data_source_filenames"])
    doc["input"] = _build_prompt(docs, question)
    doc["question"] = question
    doc["gold_answer"] = str(doc["answer"])
    return doc


def process_docs(dataset):
    """``process_docs`` hook: build the prompt for every question (all 100)."""
    return dataset.map(_add_fields)


def placeholder_metric(references, predictions) -> dict:
    return {"acc": 0.0}
