"""Response normalisation from inceptionai-abudhabi/IFEval evaluation.py (clean_for_english).

Extracted verbatim so the in-task scorer normalises responses exactly like the official eval
before running check_following. Uses only `re` + `unicodedata` (no `regex`/`absl`), so it imports
safely in every generation venv.
"""
import re
import unicodedata

_INVISIBLE_CHAR_MAP = {
    0x200B: None, 0x200C: None, 0x200D: None, 0xFEFF: None, 0x00A0: 0x20,
    0x202A: None, 0x202B: None, 0x202C: None, 0x202D: None, 0x202E: None,
    0x2066: None, 0x2067: None, 0x2068: None, 0x2069: None,
}


def clean_for_english(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE_CHAR_MAP)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf"))
    text = re.sub(r"\s+", " ", text)
    return text.strip()
