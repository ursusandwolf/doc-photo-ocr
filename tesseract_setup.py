"""Detect and install a local Windows Tesseract (no admin / no PATH edit).

Layout after install:
  %LOCALAPPDATA%/DocOCR/tesseract/tesseract.exe
  %LOCALAPPDATA%/DocOCR/tesseract/tessdata/{eng,rus}.traineddata

Path is persisted to tesseract_cmd.txt next to the app for resolve_tesseract_cmd().
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

from engines import TESSERACT_CMD_FILE, resolve_tesseract_cmd
from errors import OcrError
from i18n import t

LogFn = Callable[[str], None]

# Pinned artifacts (Windows x64). Installer is an NSIS package extractable by 7zr.
_TESSERACT_VERSION = "5.5.0.20241111"
_TESSERACT_SETUP_NAME = f"tesseract-ocr-w64-setup-{_TESSERACT_VERSION}.exe"
_TESSERACT_SETUP_URLS = (
    f"https://digi.bib.uni-mannheim.de/tesseract/{_TESSERACT_SETUP_NAME}",
    f"https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/{_TESSERACT_SETUP_NAME}",
)
_SEVEN_ZIP_URL = "https://www.7-zip.org/a/7zr.exe"
_TESSDATA_FILES = {
    "eng.traineddata": (
        "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
    ),
    "rus.traineddata": (
        "https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata"
    ),
}
_REQUIRED_LANGS = ("eng", "rus")


def install_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "DocOCR" / "tesseract"


def tessdata_dir_for(exe: Path) -> Path:
    return exe.parent / "tessdata"


def has_required_langs(exe: Path) -> bool:
    tessdata = tessdata_dir_for(exe)
    return all((tessdata / f"{code}.traineddata").is_file() for code in _REQUIRED_LANGS)


def is_tesseract_ready() -> bool:
    """True when a binary is resolvable and rus+eng traineddata are present."""
    exe = resolve_tesseract_cmd()
    if exe is None or not exe.is_file():
        return False
    return has_required_langs(exe)


def write_tesseract_cmd(exe: Path) -> None:
    TESSERACT_CMD_FILE.write_text(str(exe.resolve()) + "\n", encoding="utf-8")


def install_tesseract_local(log: Optional[LogFn] = None) -> Path:
    """Download, extract, ensure tessdata, persist tesseract_cmd.txt. Returns exe path."""
    emit = log or (lambda _message: None)
    if os.name != "nt":
        raise OcrError(t("err.tesseract_windows_only"))

    existing = resolve_tesseract_cmd()
    if existing is not None and has_required_langs(existing):
        emit(t("gui.tesseract_already", path=existing))
        write_tesseract_cmd(existing)
        return existing
    if existing is not None:
        emit(t("gui.tesseract_download_langs"))
        tessdata = tessdata_dir_for(existing)
        tessdata.mkdir(parents=True, exist_ok=True)
        for name, url in _TESSDATA_FILES.items():
            dest = tessdata / name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            _download(url, dest, emit)
        if has_required_langs(existing):
            write_tesseract_cmd(existing)
            emit(t("gui.tesseract_ready_log", path=existing))
            return existing

    target = install_dir()
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dococr-tess-") as tmp:
        tmp_dir = Path(tmp)
        setup_path = tmp_dir / _TESSERACT_SETUP_NAME
        seven_path = tmp_dir / "7zr.exe"

        emit(t("gui.tesseract_download_setup"))
        _download_first_ok(_TESSERACT_SETUP_URLS, setup_path, emit)

        emit(t("gui.tesseract_download_7zip"))
        _download(_SEVEN_ZIP_URL, seven_path, emit)

        extract_root = tmp_dir / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        emit(t("gui.tesseract_extracting"))
        _extract_nsis(seven_path, setup_path, extract_root)

        found = _find_tesseract_exe(extract_root)
        if found is None:
            raise OcrError(t("err.tesseract_extract_failed"))

        emit(t("gui.tesseract_installing", path=target))
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)
        # Copy the folder that contains tesseract.exe (and usually tessdata/).
        shutil.copytree(found.parent, target, dirs_exist_ok=True)

    exe = target / "tesseract.exe"
    if not exe.is_file():
        nested = _find_tesseract_exe(target)
        if nested is None:
            raise OcrError(t("err.tesseract_extract_failed"))
        exe = nested

    tessdata = tessdata_dir_for(exe)
    tessdata.mkdir(parents=True, exist_ok=True)
    emit(t("gui.tesseract_download_langs"))
    for name, url in _TESSDATA_FILES.items():
        dest = tessdata / name
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        _download(url, dest, emit)

    if not has_required_langs(exe):
        raise OcrError(t("err.tesseract_langs_missing"))

    write_tesseract_cmd(exe)
    emit(t("gui.tesseract_ready_log", path=exe))
    return exe


def _download_first_ok(urls: Sequence[str], dest: Path, emit: LogFn) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            _download(url, dest, emit)
            return
        except OcrError as error:
            errors.append(str(error))
    raise OcrError(t("err.tesseract_download_failed", error="; ".join(errors)))


def _download(url: str, dest: Path, emit: LogFn) -> None:
    emit(t("gui.tesseract_fetch", url=url))
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise OcrError(t("err.tesseract_download_failed", error=error)) from error
    if not data:
        raise OcrError(t("err.tesseract_download_failed", error="empty response"))
    dest.write_bytes(data)


def _extract_nsis(seven_zip: Path, archive: Path, out_dir: Path) -> None:
    command = [
        str(seven_zip),
        "x",
        "-y",
        f"-o{out_dir}",
        str(archive),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise OcrError(t("err.tesseract_extract_failed")) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise OcrError(
            t(
                "err.tesseract_extract_failed_detail",
                detail=detail or f"exit {completed.returncode}",
            )
        )


def _find_tesseract_exe(root: Path) -> Optional[Path]:
    direct = root / "tesseract.exe"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("tesseract.exe"))
    return matches[0] if matches else None
