MCQ_INSTRUCTION = (
    "السؤال التالي هو سؤال متعدد الاختيارات. اختر الإجابة الصحيحة اكتب حرف "
    "الإجابة فقط، دون أي إضافات أخرى."
)
BOOL_INSTRUCTION = 'هل المقطع التالي صحيح؟ أجب فقط بـ "نعم" أو "لا".'
CALC_INSTRUCTION = "أجب عن السؤال، مع تقديم تعليلك مسبقًا. ضع الإجابة النهائية داخل الأمر \\boxed{}."

_QUESTION_NOUN = "السؤال"
_ANSWER_NOUN = "الإجابة"
_AR_MCQ_LETTERS = ["أ", "ب", "ج"]
_YES_AR = "نعم"
_NO_AR = "لا"


def _drop_figure(dataset):
    if "figure" in dataset.column_names:
        dataset = dataset.remove_columns("figure")
    return dataset


def _relabel_choices_ar(choice_ar):
    labelled = []
    for i, line in enumerate(l for l in choice_ar.splitlines() if l.strip()):
        text = line.split(".", 1)[1].strip() if "." in line else line.strip()
        labelled.append(f"{_AR_MCQ_LETTERS[i]}. {text}")
    return labelled


def process_docs_mcq_gen(dataset):
    dataset = _drop_figure(dataset.filter(lambda d: d["task"] == "mcq"))

    def _add(doc):
        labelled = _relabel_choices_ar(doc["choice_ar"])
        doc["query"] = (
            f"{MCQ_INSTRUCTION}\n\n"
            f"{_QUESTION_NOUN}: {doc['question_ar']}\n"
            + "\n".join(labelled)
            + f"\n{_ANSWER_NOUN}:"
        )
        doc["gold_letter_ar"] = _AR_MCQ_LETTERS[ord(doc["ground_truth_ar"].strip()) - ord("A")]
        # the base mcq_prompt judge reads doc["options_with_text"]
        doc["options_with_text"] = "\n".join(labelled)
        return doc

    return dataset.map(_add)


def process_docs_bool_gen(dataset):
    dataset = _drop_figure(dataset.filter(lambda d: d["task"] == "bool"))

    def _add(doc):
        doc["query"] = f"{BOOL_INSTRUCTION}\n\n{doc['question_ar']}"
        is_yes = float(doc["ground_truth_ar"]) == 1.0
        # frame the yes/no answer as a 2-option MCQ so the base mcq_prompt can grade it:
        # the model still answers نعم/لا, the judge maps that to option أ/ب vs the gold letter.
        doc["options_with_text"] = f"{_AR_MCQ_LETTERS[0]}. {_YES_AR}\n{_AR_MCQ_LETTERS[1]}. {_NO_AR}"
        doc["gold_letter_ar"] = _AR_MCQ_LETTERS[0] if is_yes else _AR_MCQ_LETTERS[1]
        return doc

    return dataset.map(_add)


# Rows dropped because the question is broken/unanswerable from text alone
_CALC_SKIP_IDS = {
    "test_1053", "test_2715", "test_2773", "test_1088", "test_1321",
    "test_2722", "test_200", "test_2208", "test_2568", "test_2986",
    "test_2028", "test_542", "test_383", "test_700", "test_1410",
    "test_516", "test_704",
}


def process_docs_calcu(dataset):
    dataset = _drop_figure(
        dataset.filter(lambda d: d["task"] == "calcu" and d["id"] not in _CALC_SKIP_IDS)
    )

    def _add(doc):
        doc["question"] = doc["question_ar"]
        doc["input"] = f"{CALC_INSTRUCTION}\n\n{doc['question_ar']}"
        return doc

    return dataset.map(_add)


def placeholder_metric(references, predictions) -> dict:
    return {"acc": 0.0}
