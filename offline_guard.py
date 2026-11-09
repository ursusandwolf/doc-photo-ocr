"""Block accidental network use during OCR (warm + recognize).

Setup paths (pip, --download-models, Install Tesseract) must NOT enter this guard.
"""

from __future__ import annotations

import urllib.request
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def offline_session() -> Iterator[None]:
    """Patch urllib.request.urlopen so any call raises OcrError."""
    from errors import OcrError
    from i18n import t

    original = urllib.request.urlopen

    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise OcrError(t("err.offline_blocked"))

    urllib.request.urlopen = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        urllib.request.urlopen = original  # type: ignore[assignment]
