import datasets


LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def load_medarabench_subset(**kwargs):
    subset = kwargs.get("subset")
    if subset is None:
        raise ValueError("MedAraBench subset must be provided in task metadata.")

    # prepared by prepare_verified_data.py; relative to the evaluation working directory
    dataset = datasets.load_dataset(
        "parquet", data_files={"test": f"prepared_data/medarabench/{subset}/data.parquet"}
    )
    return {"test": dataset["test"]}


def doc_to_text(doc):
    choices = doc["choices"]
    allowed_letters = ", ".join(LETTER_INDICES[: len(choices)])
    option_lines = [
        f"{LETTER_INDICES[index]}:{choice}" for index, choice in enumerate(choices)
    ]

    return (
        "You are an expert medical virtual assistant.\n\n"
        f"Please provide the correct answer letter ({allowed_letters})\n\n"
        "for the following Arabic medical multiple-choice question.\n\n"
        "Question:\n\n"
        f"{doc['question']}\n\n"
        "Options:\n\n"
        + "\n\n".join(option_lines)
        + "\n\nAnswer:"
    )


def doc_to_choice(doc):
    return LETTER_INDICES[: len(doc["choices"])]


def doc_to_target(doc):
    return LETTER_INDICES[int(doc["index"])]
