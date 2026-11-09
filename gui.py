"""Simple GUI for local A4 document OCR.

1. Find Python 3.9+ (system first, otherwise runtime from the zip).
2. Install libraries into a local .venv only when missing.
3. Download models only when missing.
4. Accept a file/folder and call ocr.py through .venv.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Optional, Sequence

from i18n import get_ui_lang, set_ui_lang, t

APP_DIR = Path(__file__).resolve().parent
VENV_DIR = APP_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS = APP_DIR / "requirements.txt"
MODELS_DIR = APP_DIR / "models"
OCR_SCRIPT = APP_DIR / "ocr.py"
DEFAULT_OUTPUT = APP_DIR / "result.txt"
RUNTIME_PYTHON = APP_DIR / "runtime" / "python.exe"
UI_PREF_FILE = APP_DIR / "ui_lang.txt"
MIN_PYTHON = (3, 9)
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        set_ui_lang(load_ui_pref())
        self.geometry("720x560")
        self.minsize(640, 480)

        self.base_python = discover_base_python()
        self.output_path = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.ui_lang = tk.StringVar(value=get_ui_lang())
        self.status_python = tk.StringVar()
        self.status_libs = tk.StringVar()
        self.status_models = tk.StringVar()
        self._busy = False
        self._last_result: Optional[Path] = None
        self._last_task_key: Optional[str] = None

        self._build()
        self.apply_language()
        self.refresh_status()
        self._setup_drag_and_drop()

    def _build(self) -> None:
        self.root_frame = ttk.Frame(self, padding=12)
        self.root_frame.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(self.root_frame)
        top.pack(fill=tk.X)
        self.privacy_label = ttk.Label(top)
        self.privacy_label.pack(side=tk.LEFT, anchor=tk.W)
        lang_box = ttk.Frame(top)
        lang_box.pack(side=tk.RIGHT)
        self.lang_label = ttk.Label(lang_box)
        self.lang_label.pack(side=tk.LEFT)
        self.lang_combo = ttk.Combobox(
            lang_box,
            textvariable=self.ui_lang,
            values=("ru", "en"),
            width=5,
            state="readonly",
        )
        self.lang_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.lang_combo.bind("<<ComboboxSelected>>", self.on_language_change)

        self.status_frame = ttk.LabelFrame(self.root_frame, padding=8)
        self.status_frame.pack(fill=tk.X, pady=(10, 8))
        ttk.Label(self.status_frame, textvariable=self.status_python).pack(anchor=tk.W)
        ttk.Label(self.status_frame, textvariable=self.status_libs).pack(anchor=tk.W)
        ttk.Label(self.status_frame, textvariable=self.status_models).pack(anchor=tk.W)

        actions = ttk.Frame(self.root_frame)
        actions.pack(fill=tk.X, pady=4)
        self.btn_install = ttk.Button(actions, command=self.on_install)
        self.btn_install.pack(side=tk.LEFT)
        self.btn_models = ttk.Button(actions, command=self.on_download_models)
        self.btn_models.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_refresh = ttk.Button(actions, command=self.refresh_status)
        self.btn_refresh.pack(side=tk.LEFT, padx=(8, 0))

        self.drop_frame = ttk.LabelFrame(self.root_frame, padding=8)
        self.drop_frame.pack(fill=tk.X, pady=8)
        self.drop_label = ttk.Label(
            self.drop_frame,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            padding=18,
        )
        self.drop_label.pack(fill=tk.X)

        buttons = ttk.Frame(self.drop_frame)
        buttons.pack(fill=tk.X, pady=(8, 0))
        self.btn_file = ttk.Button(buttons, command=self.on_pick_file)
        self.btn_file.pack(side=tk.LEFT)
        self.btn_folder = ttk.Button(buttons, command=self.on_pick_folder)
        self.btn_folder.pack(side=tk.LEFT, padx=(8, 0))

        out = ttk.Frame(self.root_frame)
        out.pack(fill=tk.X, pady=4)
        self.result_label = ttk.Label(out)
        self.result_label.pack(side=tk.LEFT)
        ttk.Entry(out, textvariable=self.output_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )
        self.btn_browse_out = ttk.Button(out, text="…", width=3, command=self.on_pick_output)
        self.btn_browse_out.pack(side=tk.LEFT)
        self.btn_open = ttk.Button(out, command=self.on_open_result, state=tk.DISABLED)
        self.btn_open.pack(side=tk.LEFT, padx=(8, 0))

        self.log_frame = ttk.LabelFrame(self.root_frame, padding=4)
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = scrolledtext.ScrolledText(self.log_frame, height=12, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state=tk.DISABLED)

    def apply_language(self) -> None:
        self.title(t("gui.title"))
        self.privacy_label.configure(text=t("gui.privacy"))
        self.lang_label.configure(text=t("gui.ui_lang"))
        self.status_frame.configure(text=t("gui.status"))
        self.btn_install.configure(text=t("gui.install"))
        self.btn_models.configure(text=t("gui.models"))
        self.btn_refresh.configure(text=t("gui.refresh"))
        self.drop_frame.configure(text=t("gui.document"))
        self.drop_label.configure(text=t("gui.drop_hint"))
        self.btn_file.configure(text=t("gui.pick_file"))
        self.btn_folder.configure(text=t("gui.pick_folder"))
        self.result_label.configure(text=t("gui.result"))
        self.btn_open.configure(text=t("gui.open"))
        self.log_frame.configure(text=t("gui.log"))
        self.refresh_status()

    def on_language_change(self, _event: object = None) -> None:
        lang = set_ui_lang(self.ui_lang.get())
        self.ui_lang.set(lang)
        save_ui_pref(lang)
        self.apply_language()

    def _setup_drag_and_drop(self) -> None:
        try:
            import windnd
        except ImportError:
            self.append_log(t("gui.dnd_after_install"))
            return

        def on_drop(files: list[bytes]) -> None:
            paths = [Path(item.decode("utf-8", errors="replace")) for item in files]
            self.after(0, lambda: self.process_paths(paths))

        windnd.hook_dropfiles(self, func=on_drop)
        self.append_log(t("gui.dnd_ready"))

    def refresh_status(self) -> None:
        if self.base_python is None:
            self.status_python.set(t("gui.python_missing"))
        else:
            version = python_version_string(self.base_python)
            source = (
                t("gui.source_system")
                if is_system_python(self.base_python)
                else t("gui.source_runtime")
            )
            self.status_python.set(t("gui.python_ok", version=version, source=source))

        if libs_ready():
            self.status_libs.set(t("gui.libs_ready"))
        elif VENV_DIR.exists():
            self.status_libs.set(t("gui.libs_partial"))
        else:
            self.status_libs.set(t("gui.libs_missing"))

        if models_ready():
            self.status_models.set(t("gui.models_ready"))
        else:
            self.status_models.set(t("gui.models_missing"))

        can_run = libs_ready() and models_ready() and not self._busy
        state = tk.NORMAL if can_run else tk.DISABLED
        self.btn_file.configure(state=state)
        self.btn_folder.configure(state=state)
        self.btn_install.configure(state=tk.DISABLED if self._busy else tk.NORMAL)
        self.btn_models.configure(
            state=tk.DISABLED if self._busy or not libs_ready() else tk.NORMAL
        )

    def on_install(self) -> None:
        self._run_background("gui.task_install", self._install_libraries)

    def on_download_models(self) -> None:
        self._run_background("gui.task_models", self._download_models)

    def on_pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title=t("gui.pick_file_title"),
            filetypes=[
                (t("gui.filetype_images"), "*.jpg *.jpeg *.png *.webp *.bmp"),
                (t("gui.filetype_all"), "*.*"),
            ],
        )
        if path:
            self.process_paths([Path(path)])

    def on_pick_folder(self) -> None:
        path = filedialog.askdirectory(title=t("gui.pick_folder_title"))
        if path:
            self.process_paths([Path(path)])

    def on_pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title=t("gui.pick_output_title"),
            defaultextension=".txt",
            initialfile="result.txt",
            filetypes=[(t("gui.filetype_text"), "*.txt")],
        )
        if path:
            self.output_path.set(path)

    def on_open_result(self) -> None:
        if self._last_result and self._last_result.exists():
            os.startfile(self._last_result)  # type: ignore[attr-defined]

    def process_paths(self, paths: Sequence[Path]) -> None:
        if self._busy:
            return
        try:
            import ocr as ocr_engine

            images = ocr_engine.collect_images(paths, recursive=False)
        except Exception as error:  # noqa: BLE001
            messagebox.showwarning("OCR", str(error))
            return
        if not images:
            messagebox.showwarning("OCR", t("gui.no_images"))
            return
        output = Path(self.output_path.get().strip() or DEFAULT_OUTPUT)
        self._run_background("gui.task_ocr", lambda: self._recognize(images, output))

    def _run_background(self, title_key: str, worker: Callable[[], None]) -> None:
        if self._busy:
            return
        self._busy = True
        self._last_task_key = title_key
        title = t(title_key)
        self.refresh_status()
        self.append_log(f"--- {title} ---")

        def target() -> None:
            try:
                worker()
            except Exception as error:  # noqa: BLE001
                self.after(0, lambda: self._on_worker_error(title_key, error))
            else:
                self.after(0, lambda: self._on_worker_done(title_key))

        threading.Thread(target=target, daemon=True).start()

    def _on_worker_done(self, title_key: str) -> None:
        self._busy = False
        title = t(title_key)
        self.append_log(t("gui.done", title=title))
        self.refresh_status()
        if title_key == "gui.task_install":
            messagebox.showinfo("OCR", t("gui.install_done_body"))

    def _on_worker_error(self, title_key: str, error: Exception) -> None:
        self._busy = False
        title = t(title_key)
        self.append_log(t("gui.error", title=title, error=error))
        self.refresh_status()
        messagebox.showerror("OCR", f"{title}\n\n{error}")

    def _install_libraries(self) -> None:
        if self.base_python is None:
            raise RuntimeError(t("gui.python_required"))
        if libs_ready():
            self._log_threadsafe(t("gui.libs_already"))
            return

        if not VENV_DIR.exists():
            self._log_threadsafe(t("gui.creating_venv", python=self.base_python))
            run_checked(
                [str(self.base_python), "-m", "venv", str(VENV_DIR)],
                self._log_threadsafe,
            )

        py = str(VENV_PYTHON)
        self._log_threadsafe(t("gui.upgrade_pip"))
        run_checked([py, "-m", "pip", "install", "--upgrade", "pip"], self._log_threadsafe)

        self._log_threadsafe(t("gui.install_torch"))
        run_checked(
            [
                py,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "--index-url",
                TORCH_INDEX,
            ],
            self._log_threadsafe,
        )

        self._log_threadsafe(t("gui.install_reqs"))
        run_checked(
            [py, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            self._log_threadsafe,
        )

        if not libs_ready():
            raise RuntimeError(t("gui.libs_import_failed"))
        self._log_threadsafe(t("gui.libs_ready_log"))

    def _download_models(self) -> None:
        if not libs_ready():
            raise RuntimeError(t("gui.need_libs_first"))
        if models_ready():
            self._log_threadsafe(t("gui.models_already"))
            return

        self._log_threadsafe(t("gui.models_downloading"))
        run_checked(
            [
                str(VENV_PYTHON),
                str(OCR_SCRIPT),
                "--ui",
                get_ui_lang(),
                "--download-models",
            ],
            self._log_threadsafe,
            cwd=APP_DIR,
        )
        if not models_ready():
            raise RuntimeError(t("gui.models_missing_after"))
        self._log_threadsafe(t("gui.models_ready_log"))

    def _recognize(self, images: Sequence[Path], output: Path) -> None:
        if not libs_ready():
            raise RuntimeError(t("gui.need_libs_first"))
        if not models_ready():
            raise RuntimeError(t("gui.need_models_first"))

        command = [
            str(VENV_PYTHON),
            str(OCR_SCRIPT),
            "--ui",
            get_ui_lang(),
            "-o",
            str(output),
            *[str(path) for path in images],
        ]
        self._log_threadsafe(t("gui.files_count", count=len(images)))
        run_checked(command, self._log_threadsafe, cwd=APP_DIR)
        self._last_result = output
        self.after(0, lambda: self.btn_open.configure(state=tk.NORMAL))
        self._log_threadsafe(t("gui.saved", path=output))

    def _log_threadsafe(self, message: str) -> None:
        self.after(0, lambda: self.append_log(message))

    def append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


def load_ui_pref() -> str:
    try:
        if UI_PREF_FILE.exists():
            return set_ui_lang(UI_PREF_FILE.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    return set_ui_lang("ru")


def save_ui_pref(lang: str) -> None:
    try:
        UI_PREF_FILE.write_text(lang + "\n", encoding="utf-8")
    except OSError:
        return


def discover_base_python() -> Optional[Path]:
    candidates: list[Path] = []
    for command in (("py", "-3"), ("py",), ("python",), ("python3",)):
        resolved = resolve_command_python(command)
        if resolved is not None:
            candidates.append(resolved)
    if RUNTIME_PYTHON.exists():
        candidates.append(RUNTIME_PYTHON)

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        if python_is_compatible(path):
            return path
    return None


def resolve_command_python(command: tuple[str, ...]) -> Optional[Path]:
    try:
        completed = subprocess.run(
            [*command, "-c", "import sys; print(sys.executable)"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    executable = completed.stdout.strip()
    return Path(executable) if executable else None


def python_is_compatible(executable: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                (
                    "import sys, tkinter; "
                    "print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    parts = completed.stdout.strip().split(".")
    if len(parts) < 2:
        return False
    try:
        version = (int(parts[0]), int(parts[1]))
    except ValueError:
        return False
    return version >= MIN_PYTHON


def python_version_string(executable: Path) -> str:
    try:
        completed = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version.split()[0])"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    if completed.returncode != 0:
        return "?"
    return completed.stdout.strip() or "?"


def is_system_python(executable: Path) -> bool:
    try:
        return not (
            RUNTIME_PYTHON.exists() and executable.resolve() == RUNTIME_PYTHON.resolve()
        )
    except OSError:
        return True


def libs_ready() -> bool:
    if not VENV_PYTHON.exists():
        return False
    try:
        completed = subprocess.run(
            [str(VENV_PYTHON), "-c", "import easyocr, cv2, PIL"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def models_ready() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.glob("*.pth"))


def run_checked(
    command: Sequence[str],
    log: Callable[[str], None],
    *,
    cwd: Optional[Path] = None,
) -> None:
    log("$ " + " ".join(command))
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"exit code {code}")


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
