import datasets


LETTER_INDICES_AR = ["أ", "ب", "ج", "د"]
INSTRUCTION = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"


def load_alyah_category(**kwargs):
    category = kwargs.get("category")
    if category is None:
        raise ValueError("Alyah category must be provided in task metadata.")

    # prepared by prepare_verified_data.py; relative to the evaluation working directory
    dataset = datasets.load_dataset(
        "parquet", data_files={"test": "prepared_data/alyah/test.parquet"}
    )
    filtered_test = dataset["test"].filter(
        lambda doc: doc.get("category") == category
    )
    return {"test": filtered_test}


def doc_to_text(doc):
    query = f"{INSTRUCTION}السؤال: {doc['query']}\n"

    for index, option_key in enumerate(["option_1", "option_2", "option_3", "option_4"]):
        query += f"{LETTER_INDICES_AR[index]}) {doc[option_key]}\n"

    return query + "الإجابة:"


def doc_to_choice(doc):
    return LETTER_INDICES_AR


def doc_to_target(doc):
    return LETTER_INDICES_AR[int(doc["correct_answer"]) - 1]
