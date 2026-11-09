"""Shared user-visible OCR failures (safe across __main__ / import)."""


class OcrError(Exception):
    """User-visible failure without a traceback dump."""
