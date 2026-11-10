"""Offline OCR spell-check / auto-correct (Russian + English).

Uses pyspellchecker dictionaries shipped with the package, plus a local
office vocabulary under dicts/. No network access.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from spellchecker import SpellChecker

DICT_DIR = Path(__file__).resolve().parent / "dicts"
OFFICE_WORDS_FILE = DICT_DIR / "office_words.txt"

# Only alphabetic words are corrected; dates/codes stay in the gaps untouched.
WORD_PATTERN = re.compile(
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+(?:['’-][A-Za-zА-Яа-яЁёІіЇїЄєҐґ]+)*",
    re.UNICODE,
)

MIN_WORD_LEN = 3

# Always drop these — almost never valid in Russian office OCR.
# Guillemets «» are kept (Russian quotation marks).
# © / $ are remapped in normalize_ocr_lookalikes (©→с, $→8), not dropped.
ALWAYS_JUNK = frozenset(
    "~`^|\\<>‚‘’…�•§¤¿¡□■◆◇▪▫●○★☆►◄▲▼®™"
)

# Frequent EasyOCR/Tesseract confusions on Cyrillic office photos.
# $→8 only between word chars (1$0→180); bare $50 / 100$ stay money-like.
OCR_DOLLAR_AS_EIGHT = re.compile(r"(?<=\w)\$(?=\w)")
# ©/¢ as Cyrillic «с»: glued to letters, or a lone token (союз «с»).
# Keep «© 2024» style copyright years.
OCR_COPYRIGHT_AS_S = re.compile(
    r"(?<=[A-Za-zА-Яа-яЁё])[©¢]"
    r"|[©¢](?=[A-Za-zА-Яа-яЁё])"
    r"|(?<!\S)[©¢](?!\S)(?!\s+\d{4}\b)"
)

# EasyOCR often reads «№» as Ng / No / N° / N. before a document number.
OCR_NUMERO = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9])(?:Ng|NG|No|NO|Nо|NО|N°|Nº|N\.)(?=\s*\d)",
    re.UNICODE,
)

# Collapse mangled legal-form tokens to canonical uppercase.
OCR_LEGAL_FORM = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё])(?:ооо|oоо|оoо|ооo|OOO|Ooo)(?![A-Za-zА-Яа-яЁё])",
    re.IGNORECASE,
)
OCR_LEGAL_ZAO = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё])(?:зао|ZAO)(?![A-Za-zА-Яа-яЁё])",
    re.IGNORECASE,
)
OCR_LEGAL_PAO = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё])(?:пао|PAO)(?![A-Za-zА-Яа-яЁё])",
    re.IGNORECASE,
)
OCR_LEGAL_IP = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9])ип(?![A-Za-zА-Яа-яЁё0-9])",
    re.IGNORECASE,
)

# Dates like 15.03.2025 / 15,03,2025 / 15-03-2025 → dots.
# No spaces as separators — that glued unrelated map/coord digits into fake dates.
OCR_DATE = re.compile(
    r"\b([0-3]?\d)[./,\-]([01]?\d)[./,\-]((?:19|20)\d{2})\b"
)

# Currency / city abbreviations after OCR noise.
OCR_RUB = re.compile(r"(?<![A-Za-zА-Яа-яЁё])(?:руб|pyб|ру6)(?:\.|(?=\s|$))", re.IGNORECASE)
OCR_CITY_G = re.compile(r"(?<![A-Za-zА-Яа-яЁё])г(?:\.|(?=\s+[A-ZА-ЯЁ]))")

# If 2+ of these sit in a row, the whole run is OCR garbage
# (keeps a single № % : ; ! ? ( ) / " for real text).
NOISE_RUN_CHARS = '!"№%;:?*(~`|;\\/\'"'
NOISE_RUN = re.compile(rf"[{re.escape(NOISE_RUN_CHARS)}]{{2,}}")

# Normalize various quote styles to Russian guillemets «».
QUOTES_LOW_HIGH = re.compile(r"„([^„“\n]+)“")
QUOTES_CURLY = re.compile(r"“([^”\n]+)”")
QUOTES_STRAIGHT = re.compile(r'"([^"\n]+)"')
QUOTES_ANGLE_SPACED = re.compile(r"«\s+|\s+»")

# Lone noise symbols not glued to letters/digits.
ISOLATED_NOISE = re.compile(
    r"(?<![\w./])[*\\~`|^]+(?![\w./])",
)

# Orphan hyphen/dash after a word before space or end (год- → год).
ORPHAN_DASH = re.compile(
    r"(?<=[A-Za-zА-Яа-яЁёІіЇїЄєҐґ])[-–—](?=\s|$|[.,;:!?)\]])",
    re.UNICODE,
)
MULTI_SPACE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!])")
# "?" after a space is almost always OCR noise; "слово?" stays.
LOOSE_QUESTION = re.compile(r"\s+\?(?=\s|$)")
LOOSE_EXCLAIM = re.compile(r"\s+!(?=\s|$)")
EMPTY_PARENS = re.compile(r"\(\s*\)|\[\s*\]")
# Bare № without a following number is usually noise.
BARE_NUMERO = re.compile(r"№(?!\s*\d)")

# Protect emails / e-mail labels from junk cleanup and dictionary "fixes".
EMAIL_ADDRESS = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
)
EMAIL_LABEL = re.compile(r"(?i)\b[eе]-?mail\b")
PHONE_CHUNK = re.compile(
    r"\+?\d[\d\s\-()]{5,}\d",
)

# Always keep these office/OCR tokens even if missing from frequency lists.
WHITELIST = {
    "ооо",
    "зао",
    "пао",
    "ип",
    "hr",
    "id",
    "tab",
    "таб",
    "ng",
    "№",
    "contract",
    "г",
    "гг",
    "руб",
    "руб.",
    "pyб",
    "email",
    "e-mail",
    "е-mail",
    "mail",
}


@lru_cache(maxsize=1)
def get_spellcheckers() -> tuple[SpellChecker, SpellChecker]:
    """Load RU/EN checkers once (distance=1 for safer OCR fixes)."""
    russian = SpellChecker(language="ru", distance=1)
    english = SpellChecker(language="en", distance=1)
    extras = load_office_words()
    if extras:
        russian.word_frequency.load_words(extras)
        english.word_frequency.load_words(extras)
    russian.word_frequency.load_words(sorted(WHITELIST))
    english.word_frequency.load_words(sorted(WHITELIST))
    return russian, english


def load_office_words() -> list[str]:
    """Read local vocabulary; missing file means built-in dicts only."""
    if not OFFICE_WORDS_FILE.is_file():
        return []
    words: list[str] = []
    for raw in OFFICE_WORDS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        word = line.split()[0].lower()
        if word:
            words.append(word)
    return words


def has_unexpected_caps(token: str) -> bool:
    """True for mixed OCR junk like гОД / ПрОшу, but not ЗАЯВЛЕНИЕ or Петрова."""
    letters = [char for char in token if char.isalpha()]
    if len(letters) < 2:
        return False
    if all(char.isupper() for char in letters):
        return False
    if letters[0].isupper() and all(char.islower() for char in letters[1:]):
        return False
    return any(char.isupper() for char in letters) and any(
        char.islower() for char in letters
    )


def _mask_protected_spans(text: str) -> tuple[str, list[str]]:
    """Replace emails/phones with placeholders so cleanup cannot break them."""
    saved: list[str] = []

    def stash(match: re.Match[str]) -> str:
        saved.append(match.group(0))
        # Private-use markers — not letters, not in ALWAYS_JUNK.
        return f"\ue000{len(saved) - 1}\ue001"

    masked = EMAIL_ADDRESS.sub(stash, text)
    masked = EMAIL_LABEL.sub(stash, masked)
    masked = PHONE_CHUNK.sub(stash, masked)
    return masked, saved


def _unmask_protected_spans(text: str, saved: list[str]) -> str:
    restored = text
    for index, original in enumerate(saved):
        restored = restored.replace(f"\ue000{index}\ue001", original)
    return restored


def _strip_junk_raw(text: str) -> str:
    """Junk cleanup without email/phone masking (caller masks first)."""
    cleaned_chars: list[str] = []
    for char in text:
        if char in ALWAYS_JUNK:
            continue
        if ord(char) < 32 and char not in "\n\t":
            continue
        cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    cleaned = NOISE_RUN.sub(" ", cleaned)
    cleaned = ISOLATED_NOISE.sub("", cleaned)
    cleaned = BARE_NUMERO.sub("", cleaned)
    cleaned = ORPHAN_DASH.sub("", cleaned)
    cleaned = LOOSE_QUESTION.sub("", cleaned)
    cleaned = LOOSE_EXCLAIM.sub("", cleaned)
    cleaned = EMPTY_PARENS.sub("", cleaned)
    cleaned = SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = MULTI_SPACE.sub(" ", cleaned)
    lines = [line.strip(" \t") for line in cleaned.splitlines()]
    return "\n".join(lines).strip()


def normalize_guillemets(text: str) -> str:
    """Turn paired ASCII/curly quotes into Russian guillemets «…»."""
    if not text:
        return text
    cleaned = QUOTES_LOW_HIGH.sub(r"«\1»", text)
    cleaned = QUOTES_CURLY.sub(r"«\1»", cleaned)
    cleaned = QUOTES_STRAIGHT.sub(r"«\1»", cleaned)
    cleaned = QUOTES_ANGLE_SPACED.sub(
        lambda match: "«" if match.group(0).startswith("«") else "»",
        cleaned,
    )
    return cleaned


# Latin glyphs that OCR often emits inside Cyrillic words.
_LATIN_TO_CYR_MAP = {
    "A": "А",
    "a": "а",
    "B": "В",
    "E": "Е",
    "e": "е",
    "K": "К",
    "k": "к",
    "M": "М",
    "H": "Н",
    "O": "О",
    "o": "о",
    "P": "Р",
    "p": "р",
    "C": "С",
    "c": "с",
    "T": "Т",
    "X": "Х",
    "x": "х",
}
LATIN_LOOKALIKE_TO_CYR = str.maketrans(_LATIN_TO_CYR_MAP)
LATIN_LOOKALIKE_LETTERS = frozenset(_LATIN_TO_CYR_MAP)
# Cyrillic/Latin letters mistaken for digits inside number-like tokens.
# Only Cyrillic lookalikes → digits. Latin O/I in codes like A4-RUBEN-0O1 must stay.
_LETTER_TO_DIGIT_MAP = {
    "О": "0",
    "о": "0",
    "З": "3",
    "з": "3",
    "І": "1",
}
LETTER_LOOKALIKE_TO_DIGIT = str.maketrans(_LETTER_TO_DIGIT_MAP)
DIGIT_LOOKALIKE_LETTERS = frozenset(_LETTER_TO_DIGIT_MAP)
TOKEN_FOR_LOOKALIKE = re.compile(r"[A-Za-zА-Яа-яЁёІі0-9]+", re.UNICODE)


def _count_script_chars(token: str) -> tuple[int, int, int]:
    cyr = 0
    lat = 0
    digits = 0
    for char in token:
        if char.isdigit():
            digits += 1
        elif "А" <= char <= "я" or char in "ЁёІі":
            cyr += 1
        elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
            lat += 1
    return cyr, lat, digits


def _is_number_with_digit_lookalikes(token: str) -> bool:
    """True for tokens like 1ОЗ / 1O3 where non-digits are only 0/1/3 lookalikes."""
    if not any(char.isdigit() for char in token):
        return False
    letters = [char for char in token if not char.isdigit()]
    return bool(letters) and all(char in DIGIT_LOOKALIKE_LETTERS for char in letters)


def _latin_letters_are_only_lookalikes(token: str) -> bool:
    letters = [
        char
        for char in token
        if ("A" <= char <= "Z") or ("a" <= char <= "z")
    ]
    return bool(letters) and all(char in LATIN_LOOKALIKE_LETTERS for char in letters)


def _fix_token_script_lookalikes(token: str) -> str:
    """Fix mixed Latin/Cyrillic words and Cyrillic letters inside numbers."""
    cyr, lat, digits = _count_script_chars(token)
    if _is_number_with_digit_lookalikes(token):
        return token.translate(LETTER_LOOKALIKE_TO_DIGIT)
    # Cyrillic present + every Latin char is a lookalike (Дoгoвop), not RUBEN/ID.
    if cyr > 0 and lat > 0 and _latin_letters_are_only_lookalikes(token):
        return token.translate(LATIN_LOOKALIKE_TO_CYR)
    return token


def normalize_ocr_lookalikes(text: str) -> str:
    """Map contextual OCR glyph confusions before other polish."""
    if not text:
        return text
    cleaned = OCR_DOLLAR_AS_EIGHT.sub("8", text)
    cleaned = OCR_COPYRIGHT_AS_S.sub("с", cleaned)

    parts: list[str] = []
    last = 0
    for match in TOKEN_FOR_LOOKALIKE.finditer(cleaned):
        parts.append(cleaned[last : match.start()])
        parts.append(_fix_token_script_lookalikes(match.group(0)))
        last = match.end()
    parts.append(cleaned[last:])
    return "".join(parts)


def normalize_office_patterns(text: str) -> str:
    """Restore №, legal forms, dates, and common office abbreviations."""
    if not text:
        return text
    cleaned = OCR_NUMERO.sub("№", text)
    cleaned = OCR_LEGAL_FORM.sub("ООО", cleaned)
    cleaned = OCR_LEGAL_ZAO.sub("ЗАО", cleaned)
    cleaned = OCR_LEGAL_PAO.sub("ПАО", cleaned)
    cleaned = OCR_LEGAL_IP.sub("ИП", cleaned)
    cleaned = OCR_DATE.sub(r"\1.\2.\3", cleaned)
    cleaned = OCR_RUB.sub("руб.", cleaned)
    cleaned = OCR_CITY_G.sub("г.", cleaned)
    return cleaned


def strip_junk_text(text: str) -> str:
    """Remove OCR garbage characters; keep e-mail addresses and phone numbers."""
    if not text:
        return text
    masked, saved = _mask_protected_spans(text)
    return _unmask_protected_spans(_strip_junk_raw(masked), saved)


def correct_text(
    text: str,
    *,
    spell: bool = True,
    fix_caps: bool = True,
    strip_junk: bool = False,
    normalize: bool = True,
) -> str:
    """Polish OCR text: quotes, optional lookalikes/office/caps/dictionary fixes."""
    if not text:
        return text

    # Guillemets stay on; lookalikes/office are optional (can junk rare tokens).
    working = normalize_guillemets(text)
    if normalize:
        working = normalize_ocr_lookalikes(working)
        working = normalize_office_patterns(working)
    if not spell and not fix_caps and not strip_junk:
        return working

    masked, saved = _mask_protected_spans(working)
    working = _strip_junk_raw(masked) if strip_junk else masked

    if spell or fix_caps:
        russian, english = get_spellcheckers() if spell else (None, None)
        parts: list[str] = []
        last = 0
        for match in WORD_PATTERN.finditer(working):
            parts.append(working[last : match.start()])
            token = match.group(0)
            parts.append(
                correct_token(
                    token,
                    russian,
                    english,
                    spell=spell,
                    fix_caps=fix_caps,
                )
            )
            last = match.end()
        parts.append(working[last:])
        working = "".join(parts)

    restored = _unmask_protected_spans(working, saved)
    if strip_junk:
        restored = strip_junk_text(restored)
    return normalize_guillemets(restored)


def correct_token(
    token: str,
    russian: Optional[SpellChecker],
    english: Optional[SpellChecker],
    *,
    spell: bool,
    fix_caps: bool,
) -> str:
    working = token
    if fix_caps and has_unexpected_caps(working):
        working = working.lower()

    if not spell or russian is None or english is None:
        return working
    if len(working) < MIN_WORD_LEN:
        return working

    lower = working.lower()
    if lower in WHITELIST:
        return working
    # Keep hyphenated compounds like e-mail; dictionary often returns "email".
    if "-" in working and lower.replace("-", "") in WHITELIST:
        return working
    if lower in russian or lower in english:
        return working

    suggestion = best_correction(lower, russian, english)
    if not suggestion or suggestion == lower:
        return working
    # Do not drop hyphens from short compounds (e-mail → email).
    if "-" in working and suggestion == lower.replace("-", ""):
        return working
    return apply_case(working, suggestion)


def best_correction(
    lower: str,
    russian: SpellChecker,
    english: SpellChecker,
) -> Optional[str]:
    """Pick the better of RU/EN corrections by dictionary frequency."""
    candidates: list[tuple[int, str]] = []
    for checker in (russian, english):
        fixed = checker.correction(lower)
        if not fixed or fixed == lower:
            continue
        freq = int(checker.word_frequency.dictionary.get(fixed, 0))
        candidates.append((freq, fixed))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def apply_case(original: str, suggestion: str) -> str:
    """Match ALLCAPS / Capitalized style of the OCR token."""
    if original.isupper():
        return suggestion.upper()
    if original[0].isupper() and original[1:].islower():
        return suggestion.capitalize()
    if original[0].isupper():
        return suggestion[:1].upper() + suggestion[1:]
    return suggestion


def reset_spellcheckers_cache() -> None:
    """Test helper: clear loaded checkers."""
    get_spellcheckers.cache_clear()
