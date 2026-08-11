def load_questions(self) -> list[dict]:
    """Load LCR questions from HuggingFace dataset"""
    csv_path = hf_hub_download(
        repo_id="ArtificialAnalysis/AA-LCR",
        filename="AA-LCR_Dataset.csv",
        repo_type="dataset",
    )

    questions = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse data_source_filenames as ordered list
            if "data_source_filenames" in row and isinstance(row["data_source_filenames"], str):
                row["data_source_filenames"] = row["data_source_filenames"].split(";")

            # Parse answer as list (semicolon-separated criteria)
            if "answer" in row and isinstance(row["answer"], str):
                row["answer"] = row["answer"].split(";")
            questions.append(row)

    return questions

def get_document_set(
    self, dataset_folder: str, document_category: str, document_set_id: str, data_source_filenames: list[str]
) -> list[str]:
    """Get document set for a question in the order specified by data_source_filenames"""

    # Documents are extracted to lcr/lcr/{category}/{set_id}/ from the HuggingFace zip
    document_set_path = os.path.join(dataset_folder, document_category, document_set_id)

    document_texts = []
    for filename in data_source_filenames:
        document_path = os.path.join(document_set_path, filename)
        with open(document_path, encoding="utf-8") as f:
            document_texts.append(f.read())
    return document_texts
