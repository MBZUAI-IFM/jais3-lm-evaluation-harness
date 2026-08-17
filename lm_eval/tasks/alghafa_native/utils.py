"""AlGhafa-Native (OALL/AlGhafa-Arabic-LLM-Benchmark-Native), log-likelihood variant.

The prompt built here is byte-identical to the official OALL implementation (the instruction,
the "السؤال: " prefix, the numbered "0) option" lines and the trailing "الإجابة:"), so the
scores stay comparable to the Open Arabic LLM Leaderboard. The only change is that the choice
columns are found by matching sol1..solN in numeric order instead of "every key that is not
query/label/__few_shots", which is safer when the local parquet carries extra columns.

Used by base checkpoints; instruct/think use alghafa_native_gen.
"""

import re

INSTRUCTION = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"


def _choice_keys(doc):
    keys = [k for k in doc.keys()
            if re.fullmatch(r"sol\d+", str(k)) and doc[k] is not None]
    return sorted(keys, key=lambda k: int(k[3:]))


def process_docs(dataset):
    def _process_doc(doc):
        question = doc["query"]
        answer_index = int(str(doc["label"]).strip())
        choices = [str(doc[k]).strip() for k in _choice_keys(doc)]

        query = f"{INSTRUCTION}السؤال: {question}\n"
        for index, choice in enumerate(choices):
            query += f"{index}) {choice}\n"
        query += "الإجابة:"

        return {"query": query, "choices": choices, "gold": answer_index}

    return dataset.map(_process_doc)
