"""OCR backends behind a shared Strategy contract.

EasyOCR stays the default; Tesseract is optional (system binary + pytesseract).
Both return ocr.TextLine so preprocess/spellcheck stay engine-agnostic.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image

ENGINE_EASY = "easyocr"
ENGINE_TESSERACT = "tesseract"
ENGINE_NAMES = (ENGINE_EASY, ENGINE_TESSERACT)
DEFAULT_ENGINE = ENGINE_EASY

# EasyOCR codes -> Tesseract traineddata names.
_TESS_LANG = {
    "ru": "rus",
    "en": "eng",
    "uk": "ukr",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
}


class OcrBackend(Protocol):
    """Warm once, then recognize prepared OpenCV images into TextLine boxes."""

    name: str

    def warm(self) -> None:
        """Load models / check binaries. Called before the first recognize."""

    def recognize(
        self,
        image: np.ndarray,
        path: Any,
        *,
        on_row: Optional[Callable[[str], None]] = None,
        on_rows: Optional[Callable[[int], None]] = None,
    ) -> list[Any]:
        """Return TextLine list for one page image."""


def normalize_engine(value: str) -> str:
    name = (value or DEFAULT_ENGINE).strip().lower()
    if name not in ENGINE_NAMES:
        from errors import OcrError
        from i18n import t

        raise OcrError(t("err.bad_engine", engine=value, allowed=", ".join(ENGINE_NAMES)))
    return name


def tesseract_lang_string(languages: Sequence[str]) -> str:
    """Map EasyOCR-style codes to a Tesseract -l string (e.g. rus+eng)."""
    parts: list[str] = []
    for code in languages:
        key = code.strip().lower()
        mapped = _TESS_LANG.get(key, key)
        if mapped and mapped not in parts:
            parts.append(mapped)
    return "+".join(parts) if parts else "rus"


def create_engine(
    name: str,
    languages: Sequence[str],
    *,
    allow_download: bool = False,
) -> OcrBackend:
    engine_name = normalize_engine(name)
    if engine_name == ENGINE_TESSERACT:
        return TesseractEngine(languages)
    return EasyOcrEngine(languages, allow_download=allow_download)


def ensure_tesseract_available() -> None:
    """Fail fast with a clear message when tesseract.exe is missing."""
    from errors import OcrError
    from i18n import t

    try:
        import pytesseract
    except ModuleNotFoundError as error:
        raise OcrError(t("err.no_pytesseract")) from error

    try:
        pytesseract.get_tesseract_version()
    except Exception as error:  # noqa: BLE001
        raise OcrError(t("err.no_tesseract")) from error


def parse_tesseract_data(data: dict[str, Any]) -> list[Any]:
    """Convert pytesseract.image_to_data(Output.DICT) into TextLine values."""
    from ocr import TextLine

    texts = data.get("text") or []
    lines: list[Any] = []
    count = len(texts)
    for index in range(count):
        raw = texts[index]
        cleaned = str(raw).strip() if raw is not None else ""
        if not cleaned:
            continue
        try:
            conf_raw = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            conf_raw = -1.0
        confidence = 0.0 if conf_raw < 0 else min(conf_raw / 100.0, 1.0)
        try:
            left = int(data["left"][index])
            top = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
        except (KeyError, TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        box = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        )
        lines.append(TextLine(text=cleaned, confidence=confidence, box=box))
    return lines


class EasyOcrEngine:
    """CRAFT detect + CRNN recognize via the existing ocr.read_lines path."""

    name = ENGINE_EASY

    def __init__(self, languages: Sequence[str], *, allow_download: bool = False) -> None:
        self._languages = list(languages)
        self._allow_download = allow_download
        self._reader: Any = None

    def warm(self) -> None:
        from ocr import create_reader

        self._reader = create_reader(
            self._languages,
            allow_download=self._allow_download,
        )

    def recognize(
        self,
        image: np.ndarray,
        path: Any,
        *,
        on_row: Optional[Callable[[str], None]] = None,
        on_rows: Optional[Callable[[int], None]] = None,
    ) -> list[Any]:
        from ocr import read_lines

        if self._reader is None:
            self.warm()
        assert self._reader is not None
        return read_lines(
            self._reader,
            image,
            path,
            on_row=on_row,
            on_rows=on_rows,
        )


class TesseractEngine:
    """System Tesseract via pytesseract; boxes come from image_to_data."""

    name = ENGINE_TESSERACT

    def __init__(self, languages: Sequence[str]) -> None:
        self._languages = list(languages)
        self._lang = tesseract_lang_string(languages)

    def warm(self) -> None:
        ensure_tesseract_available()

    def recognize(
        self,
        image: np.ndarray,
        path: Any,
        *,
        on_row: Optional[Callable[[str], None]] = None,
        on_rows: Optional[Callable[[int], None]] = None,
    ) -> list[Any]:
        from errors import OcrError
        from i18n import t
        from ocr import box_left, group_lines_into_rows

        ensure_tesseract_available()
        import pytesseract

        try:
            rgb = _to_rgb_uint8(image)
            pil_image = Image.fromarray(rgb)
            data = pytesseract.image_to_data(
                pil_image,
                lang=self._lang,
                output_type=pytesseract.Output.DICT,
            )
            words = parse_tesseract_data(data)
            if not words:
                if on_rows is not None:
                    on_rows(0)
                return []

            rows = group_lines_into_rows(words)
            if on_rows is not None:
                on_rows(len(rows))
            ordered: list[Any] = []
            for row in rows:
                row_lines = sorted(row, key=box_left)
                ordered.extend(row_lines)
                if on_row is not None:
                    row_text = " ".join(
                        part.text for part in row_lines if part.text
                    ).strip()
                    on_row(row_text)
            return ordered
        except OcrError:
            raise
        except Exception as error:  # noqa: BLE001
            raise OcrError(t("err.recognize", name=path.name, error=error)) from error


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image
