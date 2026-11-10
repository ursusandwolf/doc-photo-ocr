"""Reading-order layout modes shared by GUI, CLI, and OCR engines.

Kept free of cv2/EasyOCR imports so the GUI can load prefs without warming OCR.
"""

from __future__ import annotations

LAYOUT_PLAIN = "plain"
LAYOUT_TABLE = "table"
LAYOUT_NAMES = (LAYOUT_PLAIN, LAYOUT_TABLE)
DEFAULT_LAYOUT = LAYOUT_PLAIN


def normalize_layout(value: str) -> str:
    name = (value or DEFAULT_LAYOUT).strip().lower()
    if name not in LAYOUT_NAMES:
        from errors import OcrError
        from i18n import t

        raise OcrError(t("err.bad_layout", layout=value, allowed=", ".join(LAYOUT_NAMES)))
    return name
