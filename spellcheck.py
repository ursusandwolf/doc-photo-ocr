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


def correct_text(text: str) -> str:
    """Correct unknown alphabetic words; keep dates, codes, punctuation as-is."""
    if not text:
        return text
    russian, english = get_spellcheckers()
    parts: list[str] = []
    last = 0
    for match in WORD_PATTERN.finditer(text):
        parts.append(text[last : match.start()])
        parts.append(correct_token(match.group(0), russian, english))
        last = match.end()
    parts.append(text[last:])
    return "".join(parts)


def correct_token(token: str, russian: SpellChecker, english: SpellChecker) -> str:
    if len(token) < MIN_WORD_LEN:
        return token

    lower = token.lower()
    if lower in WHITELIST:
        return token
    if lower in russian or lower in english:
        return token

    suggestion = best_correction(lower, russian, english)
    if not suggestion or suggestion == lower:
        return token
    return apply_case(token, suggestion)


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
