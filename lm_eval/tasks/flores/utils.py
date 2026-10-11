import os
from functools import lru_cache

import datasets

FLORES_REPO = "openlanguagedata/flores_plus"
# pinned: verified to equal the former local devtest.csv for all 10 languages x 1,012 ids
FLORES_REVISION = "e707e62e762a9e2faca3933aa315f4665cc3cd8f"
FLORES_LANGUAGES = (
    "eng_Latn",
    "acm_Arab",
    "acq_Arab",
    "aeb_Arab",
    "apc_Arab_nort3139",
    "apc_Arab_sout3123",
    "arb_Arab",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
)


@lru_cache(maxsize=1)
def _load_devtest():
    # gated dataset: the token comes from HF_TOKEN (eval.py --hf-creds-path)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None
    columns = {}
    for code in FLORES_LANGUAGES:
        split = datasets.load_dataset(
            FLORES_REPO,
            code,
            split="devtest",
            revision=FLORES_REVISION,
            token=token,
        )
        if "id" not in columns:
            columns["id"] = list(split["id"])
        elif list(split["id"]) != columns["id"]:
            raise ValueError(f"FLORES+ ids of `{code}` do not align with eng_Latn")
        columns[f"{code}_sentence"] = list(split["text"])
    return datasets.Dataset.from_dict(columns)


def load_flores_devtest(**kwargs):
    """Wide devtest table (id, <code>_sentence ...), the layout of the former devtest.csv."""
    return {"test": _load_devtest()}
