"""Arabic EXAMS (OALL/Arabic_EXAMS), log-likelihood variant.

The prompt is byte-identical to the official OALL implementation
(lm_eval/tasks/arabic_leaderboard_complete/arabic_leaderboard_arabic_exams): the per-subject
instruction, the "السؤال: " prefix, the " أ) choice\\n" option lines joined by "\\n" (which is
why the options end up double-spaced -- kept deliberately), and the trailing "الإجابة:".
Choices are the Arabic option LETTERS, not the option texts, so the log-likelihood comparison is
over أ/ب/ج/د exactly as OALL scores it.

Named `exams_ar` rather than `arabic_exams` because the upstream task of that name is already
registered in this harness and duplicate task names get silently skipped.

Used by base checkpoints; instruct/think use exams_ar_gen.
"""

# fmt: off
LETTER_INDICES_AR = ["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح", "ط", "ي"]
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
# fmt: on

CHOICE_KEYS = ["A", "B", "C", "D"]


def process_docs(dataset):
    def _process_doc(doc):
        topic = doc["subject"]
        question = doc["question"]
        choices = [doc[k] for k in CHOICE_KEYS]
        choices_formatted = [
            f" {LETTER_INDICES_AR[i]}) {choice}\n" for i, choice in enumerate(choices)
        ]
        answer = doc["answer"]
        answer_index = LETTER_INDICES.index(answer)

        instruction = (
            "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح حول "
            f"{topic.replace('_', ' ')}. \n\n"
        )
        query = f"{instruction}السؤال: {question}\n"
        query += "\n".join(choices_formatted)
        query += "\nالإجابة:"

        return {
            "query": query,
            "choices": LETTER_INDICES_AR[: len(CHOICE_KEYS)],
            "gold": answer_index,
        }

    return dataset.map(_process_doc)
