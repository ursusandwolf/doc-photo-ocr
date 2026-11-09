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
OCR_LOOKALIKES = str.maketrans(
    {
        "©": "с",  # conjunction «с» / letter с
        "¢": "с",
        "$": "8",
    }
)

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


def normalize_ocr_lookalikes(text: str) -> str:
    """Map common OCR glyph confusions (©→с, $→8) before other polish."""
    if not text:
        return text
    return text.translate(OCR_LOOKALIKES)


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
) -> str:
    """Polish OCR text: quotes, optional junk/caps/dictionary fixes."""
    if not text:
        return text

    # Guillemets + lookalikes always — they fix structure, not optional polish.
    working = normalize_guillemets(text)
    working = normalize_ocr_lookalikes(working)
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
