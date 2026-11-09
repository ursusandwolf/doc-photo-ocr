"""Collect image file paths for CLI and GUI without importing OpenCV/OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class PathError(Exception):
    """Invalid input path. key maps to an i18n message id."""

    def __init__(self, key: str, **kwargs: Any) -> None:
        self.key = key
        self.kwargs = kwargs
        super().__init__(key)


def collect_images(paths: Sequence[Path], *, recursive: bool = False) -> list[Path]:
    """Collect unique images from files and folders in stable order."""
    images: list[Path] = []
    for path in paths:
        if not path.exists():
            raise PathError("err.path_missing", path=path)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(path)
            else:
                ext = path.suffix.lower() or "none"
                raise PathError(
                    "err.bad_format",
                    ext=ext,
                    allowed=", ".join(sorted(SUPPORTED_EXTENSIONS)),
                )
            continue
        if path.is_dir():
            images.extend(list_images_in_directory(path, recursive=recursive))
            continue
        raise PathError("err.not_file_or_dir", path=path)

    unique: list[Path] = []
    seen: set[str] = set()
    for image in images:
        key = str(image.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(image)
    return unique


def list_images_in_directory(directory: Path, *, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    found = [
        child
        for child in iterator
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(found, key=lambda item: str(item).lower())
