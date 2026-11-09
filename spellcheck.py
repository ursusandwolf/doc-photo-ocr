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
ALWAYS_JUNK = frozenset(
    "~`^|\\<>«»„“”‚‘’…�•§¤¿¡□■◆◇▪▫●○★☆►◄▲▼©®™"
)

# If 2+ of these sit in a row, the whole run is OCR garbage
# (keeps a single № % : ; ! ? ( ) / " for real text).
NOISE_RUN_CHARS = '!"№%;:?*(~`|;\\/„“”«»\'"'
NOISE_RUN = re.compile(rf"[{re.escape(NOISE_RUN_CHARS)}]{{2,}}")

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


def strip_junk_text(text: str) -> str:
    """Remove OCR garbage characters and tidy leftover spacing/punctuation."""
    if not text:
        return text
    cleaned_chars: list[str] = []
    for char in text:
        if char in ALWAYS_JUNK:
            continue
        if ord(char) < 32 and char not in "\n\t":
            continue
        cleaned_chars.append(char)
    cleaned = "".join(cleaned_chars)
    # Drop punctuation clusters like !"№;%::?*( first.
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


def correct_text(
    text: str,
    *,
    spell: bool = True,
    fix_caps: bool = True,
    strip_junk: bool = True,
) -> str:
    """Polish OCR text: junk chars, weird caps, dictionary fixes."""
    if not text:
        return text
    if not spell and not fix_caps and not strip_junk:
        return text

    working = strip_junk_text(text) if strip_junk else text
    if not spell and not fix_caps:
        return working

    russian, english = get_spellcheckers() if spell else (None, None)
    parts: list[str] = []
    last = 0
    for match in WORD_PATTERN.finditer(working):
        parts.append(working[last : match.start()])
        parts.append(
            correct_token(
                match.group(0),
                russian,
                english,
                spell=spell,
                fix_caps=fix_caps,
            )
        )
        last = match.end()
    parts.append(working[last:])
    polished = "".join(parts)
    # Second pass: caps/spell may leave spaces before punctuation tidy again.
    return strip_junk_text(polished) if strip_junk else polished


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
    if lower in russian or lower in english:
        return working

    suggestion = best_correction(lower, russian, english)
    if not suggestion or suggestion == lower:
        return working
    # After caps-normalization prefer lowercase unless the token stayed titled/ALLCAPS.
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
