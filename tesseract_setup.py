"""Detect and install a local Windows Tesseract (user folder + one UAC prompt).

Layout after install:
  %LOCALAPPDATA%/DocOCR/tesseract/tesseract.exe
  %LOCALAPPDATA%/DocOCR/tesseract/tessdata/{eng,rus}.traineddata

Path is persisted to tesseract_cmd.txt next to the app for resolve_tesseract_cmd().

The UB-Mannheim setup is an NSIS installer that cannot be unpacked by 7-Zip as an
archive and requires elevation even for a custom directory — so we run it silently
with ShellExecute \"runas\" into LocalAppData.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Sequence

from engines import TESSERACT_CMD_FILE, resolve_tesseract_cmd
from errors import OcrError
from i18n import t

LogFn = Callable[[str], None]

# Pinned UB-Mannheim Windows x64 NSIS installer (official GitHub release asset).
_TESSERACT_VERSION = "5.5.3.20260724"
_TESSERACT_TAG = "5.5.3"
_TESSERACT_SETUP_NAME = f"tesseract-ocr-w64-setup-{_TESSERACT_VERSION}.exe"
_TESSERACT_SETUP_URLS = (
    f"https://github.com/tesseract-ocr/tesseract/releases/download/"
    f"{_TESSERACT_TAG}/{_TESSERACT_SETUP_NAME}",
)
# SHA-256 of the exact pinned bytes (recompute when bumping the version).
_TESSERACT_SETUP_SHA256 = (
    "bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4"
)
_TESSDATA_FILES = {
    "eng.traineddata": (
        "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata",
        "daa0c97d651c19fba3b25e81317cd697e9908c8208090c94c3905381c23fc047",
    ),
    "rus.traineddata": (
        "https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata",
        "681be2c2bead1bc7bd235df88c44e8e60ae73ae866840c0ad4e3b4c247bd37c2",
    ),
}
_REQUIRED_LANGS = ("eng", "rus")
_INSTALL_WAIT_SECONDS = 600


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
    """Download installer, silent-install to LocalAppData, ensure tessdata, persist path."""
    emit = log or (lambda _message: None)
    if os.name != "nt":
        raise OcrError(t("err.tesseract_windows_only"))

    existing = resolve_tesseract_cmd()
    if existing is not None and has_required_langs(existing):
        emit(t("gui.tesseract_already", path=existing))
        write_tesseract_cmd(existing)
        return existing
    if existing is not None:
        _ensure_langs(existing, emit)
        if has_required_langs(existing):
            write_tesseract_cmd(existing)
            emit(t("gui.tesseract_ready_log", path=existing))
            return existing

    target = install_dir()
    with tempfile.TemporaryDirectory(prefix="dococr-tess-") as tmp:
        tmp_dir = Path(tmp)
        setup_path = tmp_dir / _TESSERACT_SETUP_NAME
        emit(t("gui.tesseract_download_setup"))
        _download_first_ok(_TESSERACT_SETUP_URLS, setup_path, emit)

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True, exist_ok=True)

        emit(t("gui.tesseract_installing", path=target))
        emit(t("gui.tesseract_uac_hint"))
        _silent_install_nsis(setup_path, target)

    exe = target / "tesseract.exe"
    if not exe.is_file():
        nested = _find_tesseract_exe(target)
        if nested is None:
            raise OcrError(t("err.tesseract_install_failed"))
        exe = nested

    _ensure_langs(exe, emit)
    if not has_required_langs(exe):
        raise OcrError(t("err.tesseract_langs_missing"))

    write_tesseract_cmd(exe)
    emit(t("gui.tesseract_ready_log", path=exe))
    return exe


def _ensure_langs(exe: Path, emit: LogFn) -> None:
    tessdata = tessdata_dir_for(exe)
    tessdata.mkdir(parents=True, exist_ok=True)
    emit(t("gui.tesseract_download_langs"))
    for name, (url, digest) in _TESSDATA_FILES.items():
        dest = tessdata / name
        if dest.is_file() and _file_sha256(dest) == digest:
            continue
        _download(url, dest, emit, expected_sha256=digest)


def _download_first_ok(urls: Sequence[str], dest: Path, emit: LogFn) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            _download(url, dest, emit, expected_sha256=_TESSERACT_SETUP_SHA256)
            return
        except OcrError as error:
            errors.append(str(error))
    raise OcrError(t("err.tesseract_download_failed", error="; ".join(errors)))


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_sha256(path: Path, expected: str) -> None:
    actual = _file_sha256(path)
    if actual.lower() != expected.lower():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise OcrError(
            t(
                "err.tesseract_checksum",
                name=path.name,
                expected=expected[:16],
                actual=actual[:16],
            )
        )


def _download(
    url: str,
    dest: Path,
    emit: LogFn,
    *,
    expected_sha256: str,
) -> None:
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
    _verify_sha256(dest, expected_sha256)
    emit(t("gui.tesseract_checksum_ok", name=dest.name))


def _silent_install_nsis(setup: Path, target: Path) -> None:
    """Run UB-Mannheim NSIS setup elevated: /S /D=<target> ( /D must be last )."""
    # No quotes around /D= path — NSIS requirement.
    params = f"/S /D={target}"
    try:
        _run_elevated(setup, params)
    except OcrError:
        raise
    except OSError as error:
        raise OcrError(t("err.tesseract_install_failed")) from error

    deadline = time.monotonic() + _INSTALL_WAIT_SECONDS
    while time.monotonic() < deadline:
        if _find_tesseract_exe(target) is not None:
            return
        time.sleep(0.5)
    raise OcrError(t("err.tesseract_install_failed"))


def _run_elevated(executable: Path, parameters: str) -> None:
    """ShellExecuteExW runas + wait. User must accept the UAC prompt."""
    import ctypes
    from ctypes import wintypes

    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    SW_HIDE = 0
    INFINITE = 0xFFFFFFFF
    WAIT_FAILED = 0xFFFFFFFF

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hKeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(info)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = str(executable)
    info.lpParameters = parameters
    info.nShow = SW_HIDE

    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):
        raise OcrError(t("err.tesseract_uac_denied"))
    if not info.hProcess:
        raise OcrError(t("err.tesseract_uac_denied"))

    waited = ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, INFINITE)
    if waited == WAIT_FAILED:
        ctypes.windll.kernel32.CloseHandle(info.hProcess)
        raise OcrError(t("err.tesseract_install_failed"))

    code = wintypes.DWORD()
    ctypes.windll.kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(info.hProcess)
    if code.value not in (0, 3010):  # 3010 = success, reboot required
        raise OcrError(t("err.tesseract_install_exit", code=code.value))


def _find_tesseract_exe(root: Path) -> Optional[Path]:
    direct = root / "tesseract.exe"
    if direct.is_file():
        return direct
    matches = sorted(root.rglob("tesseract.exe"))
    return matches[0] if matches else None
