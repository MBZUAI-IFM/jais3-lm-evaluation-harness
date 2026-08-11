import datasets
import evaluate


def strip(resps, docs):
    """
    Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
    """
    return map(lambda r: r[0].strip(), resps)


def dr_ar(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "dr_ar")


def ar_dr(dataset: datasets.Dataset):
    return dataset.filter(lambda x: x["direction"] == "ar_dr")


def doc_to_text(doc):
    additional_instruction = "عافاك، كتب غير المطلوب، بلا حتى مقدمة، ولا تعليق، ولا شرح، ولا حتى كلام زايد."
    if doc["direction"] == "dr_ar":
        arabizi_confusion_instruction = "العربزي تعني arabizi."
        additional_instruction = f"{additional_instruction}\n{arabizi_confusion_instruction}"
    doc_text = doc["messages"][0]["content"]
    return f"{additional_instruction}\n{doc_text}"


def doc_to_target(doc):
    return doc["messages"][1]["content"]


def bert(items):
    return items


def Average(lst):
    return sum(lst) / len(lst)


def arabizibert(items):
    bert_model = "SI2M-Lab/DarijaBERT-arabizi"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )
    return Average(bert["f1"])


def darijabert(items):
    bert_model = "SI2M-Lab/DarijaBERT"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )
    return Average(bert["f1"])


def mbert(items):
    bert_model = "google-bert/bert-base-multilingual-cased"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items)
    bert = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )
    return Average(bert["f1"])
