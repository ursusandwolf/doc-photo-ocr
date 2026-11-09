"""Simple GUI for local A4 document OCR.

1. Show the window immediately.
2. Find Python 3.9+ (system first, otherwise runtime from the zip).
3. Install libraries into a local .venv only when missing.
4. Download models only when missing.
5. Warm a long-lived OCR worker (EasyOCR stays in memory) with a progress bar.
6. Recognize files through that worker.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Optional, Sequence, Union

from PIL import Image, ImageGrab

from engines import (
    DEFAULT_ENGINE,
    ENGINE_EASY,
    ENGINE_NAMES,
    ENGINE_TESSERACT,
    normalize_engine,
    resolve_tesseract_cmd,
)
from i18n import get_ui_lang, set_ui_lang, t
from image_paths import PathError as ImagePathError
from image_paths import collect_images
from ocr import DEFAULT_LAYOUT, LAYOUT_NAMES, normalize_layout
from tesseract_setup import (
    install_tesseract_local,
    is_tesseract_ready,
    tesseract_trust_info,
)


APP_DIR = Path(__file__).resolve().parent
VENV_DIR = APP_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS = APP_DIR / "requirements.txt"
MODELS_DIR = APP_DIR / "models"
OCR_SCRIPT = APP_DIR / "ocr.py"
DEFAULT_OUTPUT = APP_DIR / "result.txt"
RUNTIME_PYTHON = APP_DIR / "runtime" / "python.exe"
UI_PREF_FILE = APP_DIR / "ui_lang.txt"
THEME_PREF_FILE = APP_DIR / "ui_theme.txt"
ENGINE_PREF_FILE = APP_DIR / "ui_engine.txt"
LAYOUT_PREF_FILE = APP_DIR / "ui_layout.txt"
DEFAULT_THEME = "light"
THEME_NAMES = ("light", "gray", "dark")
# Tk theming tokens (clam). Wait label stays red on every theme.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "bg": "#f0f0f0",
        "fg": "#1a1a1a",
        "field": "#ffffff",
        "button": "#e4e4e4",
        "button_active": "#d0d0d0",
        "button_disabled": "#bdbdbd",
        "button_disabled_fg": "#424242",
        "accent": "#1565c0",
        "trough": "#cfcfcf",
        "select": "#c5d8f0",
        "wait": "#c62828",
        "ok": "#2e7d32",
        "bad": "#c62828",
    },
    "gray": {
        "bg": "#5a5a5a",
        "fg": "#f2f2f2",
        "field": "#454545",
        "button": "#6e6e6e",
        "button_active": "#7a7a7a",
        "button_disabled": "#8a8a8a",
        "button_disabled_fg": "#f2f2f2",
        "accent": "#90caf9",
        "trough": "#3a3a3a",
        "select": "#607d8b",
        "wait": "#ff8a80",
        "ok": "#81c784",
        "bad": "#ff8a80",
    },
    "dark": {
        "bg": "#1e1e1e",
        "fg": "#e6e6e6",
        "field": "#2b2b2b",
        "button": "#3a3a3a",
        "button_active": "#4a4a4a",
        "button_disabled": "#6a6a6a",
        "button_disabled_fg": "#e6e6e6",
        "accent": "#64b5f6",
        "trough": "#121212",
        "select": "#37474f",
        "wait": "#ef5350",
        "ok": "#81c784",
        "bad": "#ef5350",
    },
}
MIN_PYTHON = (3, 9)
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
_LIBS_READY_CACHE: Optional[bool] = None

class OcrEngine:
    """Long-lived `ocr.py --worker` process with one OCR backend already loaded."""

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self.ready = False
        self.engine_name = DEFAULT_ENGINE

    def start(
        self,
        *,
        engine: str,
        on_progress: Callable[[int, int], None],
    ) -> None:
        self.stop()
        if not VENV_PYTHON.exists():
            raise RuntimeError(t("gui.need_libs_first"))
        engine_name = normalize_engine(engine)
        if engine_name == ENGINE_EASY and not models_ready():
            raise RuntimeError(t("gui.need_models_first"))

        command = [
            str(VENV_PYTHON),
            str(OCR_SCRIPT),
            "--ui",
            get_ui_lang(),
            "--engine",
            engine_name,
            "--worker",
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(APP_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **hidden_subprocess_kwargs(),
        )
        process = self._process
        assert process is not None and process.stdout is not None

        stray_lines: list[str] = []
        while True:
            if self._process is not process:
                raise RuntimeError("OCR worker was stopped")
            try:
                line = process.stdout.readline()
            except (OSError, ValueError) as error:
                raise RuntimeError("OCR worker was stopped") from error
            if not line:
                code = process.poll()
                detail = "\n".join(stray_lines).strip()
                if detail:
                    raise RuntimeError(detail)
                raise RuntimeError(f"OCR worker exited early (code {code})")
            line = line.rstrip("\n")
            if line.startswith("##PROGRESS "):
                payload = line[len("##PROGRESS ") :].strip()
                current, total = parse_progress_fraction(payload)
                on_progress(current, total)
            elif line.startswith("##READY"):
                self.ready = True
                self.engine_name = engine_name
                return
            elif line.startswith("##ERROR "):
                raise RuntimeError(line[len("##ERROR ") :].strip())
            elif line.strip():
                stray_lines.append(line.strip())

    def recognize(
        self,
        images: Sequence[Path],
        output: Path,
        *,
        ui: str,
        spell: bool = True,
        fix_caps: bool = True,
        strip_junk: bool = False,
        invert: bool = False,
        layout: str = DEFAULT_LAYOUT,
        on_file: Callable[[int, int, str], None],
        on_line: Optional[Callable[[str], None]] = None,
        on_rows: Optional[Callable[[int], None]] = None,
    ) -> None:
        if not self.ready or self._process is None:
            raise RuntimeError(t("gui.engine_not_ready"))
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(t("gui.engine_not_ready"))

        request = {
            "cmd": "recognize",
            "images": [str(path) for path in images],
            "output": str(output),
            "ui": ui,
            "spell": spell,
            "fix_caps": fix_caps,
            "strip_junk": strip_junk,
            "invert": invert,
            "layout": normalize_layout(layout),
        }
        with self._lock:
            self._process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self._process.stdin.flush()
            while True:
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("OCR worker stopped during recognition")
                line = line.rstrip("\n")
                if line.startswith("##FILE "):
                    payload = line[len("##FILE ") :].strip()
                    current, total, path = parse_file_progress(payload)
                    on_file(current, total, path)
                elif line.startswith("##ROWS "):
                    if on_rows is not None:
                        try:
                            on_rows(int(line[len("##ROWS ") :].strip()))
                        except ValueError:
                            on_rows(0)
                elif line.startswith("##LINE "):
                    if on_line is not None:
                        on_line(line[len("##LINE ") :])
                elif line.startswith("##LINE"):
                    if on_line is not None:
                        on_line("")
                elif line.startswith("##OK"):
                    return
                elif line.startswith("##ERROR "):
                    raise RuntimeError(line[len("##ERROR ") :].strip())

    def stop(self) -> None:
        self.ready = False
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                process.stdin.write('{"cmd":"quit"}\n')
                process.stdin.flush()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def parse_progress_fraction(payload: str) -> tuple[int, int]:
    """Parse '2/3 loading' into (2, 3)."""
    token = payload.split()[0] if payload else "0/1"
    if "/" not in token:
        return 0, 1
    left, right = token.split("/", 1)
    try:
        return int(left), max(1, int(right))
    except ValueError:
        return 0, 1


def parse_file_progress(payload: str) -> tuple[int, int, str]:
    """Parse '1/3 path.png' into (1, 3, 'path.png')."""
    parts = payload.split(maxsplit=1)
    current, total = parse_progress_fraction(parts[0] if parts else "0/1")
    path = parts[1] if len(parts) > 1 else ""
    return current, total, path


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        set_ui_lang(load_ui_pref())
        self.geometry("720x560")
        self.minsize(640, 480)

        # Heavy discovery/warmup happens after the window is visible.
        self.base_python: Optional[Path] = None
        self.output_path = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.spell_enabled = tk.BooleanVar(value=True)
        self.fix_caps_enabled = tk.BooleanVar(value=True)
        self.strip_junk_enabled = tk.BooleanVar(value=False)
        self.invert_enabled = tk.BooleanVar(value=False)
        self.ui_lang = tk.StringVar(value=get_ui_lang())
        self.ui_theme = tk.StringVar(value=load_theme_pref())
        self.ui_engine = tk.StringVar(value=load_engine_pref())
        self.ui_layout = tk.StringVar(value=load_layout_pref())
        self.status_line = tk.StringVar()
        self.progress_value = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="")
        self._busy = False
        self._engine_starting = False
        self._engine_ready = False
        self._engine = OcrEngine()
        self._wait_animating = False
        self._wait_message_key = "gui.wait_ocr"
        self._wait_detail = ""
        self._wait_started_at = 0.0
        self._row_done = 0
        self._row_total = 0
        self._last_result: Optional[Path] = None
        self._last_task_key: Optional[str] = None
        self._last_good_engine = ENGINE_EASY
        self._engine_start_gen = 0
        # Drop callback runs inside a Win32 WndProc — never call Tk from there.
        self._drop_queue: queue.Queue[Union[list[Path], BaseException]] = queue.Queue()
        # Worker thread must not call Tk; main loop drains this queue.
        self._engine_ui_queue: queue.Queue[tuple] = queue.Queue()

        self._build()
        self.apply_theme()
        self.apply_language()
        self._paint_status_placeholder()
        self._setup_drag_and_drop()
        self.bind("<Control-v>", self.on_paste_image)
        self.bind("<Control-V>", self.on_paste_image)
        self.bind("<Shift-Insert>", self.on_paste_image)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self._boot)
        self.after(200, self._poll_drop_queue)
        self.after(50, self._poll_engine_ui_queue)

    def _build(self) -> None:
        self.root_frame = ttk.Frame(self, padding=12)
        self.root_frame.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(self.root_frame)
        top.pack(fill=tk.X)
        self.privacy_label = ttk.Label(top)
        self.privacy_label.pack(side=tk.LEFT, anchor=tk.W)
        lang_box = ttk.Frame(top)
        lang_box.pack(side=tk.RIGHT)
        self.theme_label = ttk.Label(lang_box)
        self.theme_label.pack(side=tk.LEFT)
        self.theme_combo = ttk.Combobox(
            lang_box,
            textvariable=self.ui_theme,
            values=THEME_NAMES,
            width=7,
            state="readonly",
        )
        self.theme_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.theme_combo.bind("<<ComboboxSelected>>", self.on_theme_change)
        self.engine_label = ttk.Label(lang_box)
        self.engine_label.pack(side=tk.LEFT)
        self.engine_combo = ttk.Combobox(
            lang_box,
            textvariable=self.ui_engine,
            values=ENGINE_NAMES,
            width=10,
            state="readonly",
        )
        self.engine_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.engine_combo.bind("<<ComboboxSelected>>", self.on_engine_change)
        self.layout_label = ttk.Label(lang_box)
        self.layout_label.pack(side=tk.LEFT)
        self.layout_combo = ttk.Combobox(
            lang_box,
            textvariable=self.ui_layout,
            values=LAYOUT_NAMES,
            width=7,
            state="readonly",
        )
        self.layout_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.layout_combo.bind("<<ComboboxSelected>>", self.on_layout_change)
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

        # Bottom status bar is packed first (side=BOTTOM) so it stays one line.
        self.status_bar = ttk.Label(
            self.root_frame,
            textvariable=self.status_line,
            style="StatusBad.TLabel",
            anchor=tk.W,
            padding=(2, 4),
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        actions = ttk.Frame(self.root_frame)
        actions.pack(fill=tk.X, pady=(10, 4))
        self.btn_install = ttk.Button(actions, command=self.on_install)
        self.btn_install.pack(side=tk.LEFT)
        self.btn_models = ttk.Button(actions, command=self.on_download_models)
        self.btn_models.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_tesseract = ttk.Button(actions, command=self.on_install_tesseract)
        self.btn_tesseract.pack(side=tk.LEFT, padx=(8, 0))
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
        self.spell_check = ttk.Checkbutton(
            buttons,
            variable=self.spell_enabled,
        )
        self.spell_check.pack(side=tk.LEFT, padx=(12, 0))
        self.fix_caps_check = ttk.Checkbutton(
            buttons,
            variable=self.fix_caps_enabled,
        )
        self.fix_caps_check.pack(side=tk.LEFT, padx=(12, 0))
        self.strip_junk_check = ttk.Checkbutton(
            buttons,
            variable=self.strip_junk_enabled,
        )
        self.strip_junk_check.pack(side=tk.LEFT, padx=(12, 0))
        self.invert_check = ttk.Checkbutton(
            buttons,
            variable=self.invert_enabled,
        )
        self.invert_check.pack(side=tk.LEFT, padx=(12, 0))

        out = ttk.Frame(self.root_frame)
        out.pack(fill=tk.X, pady=4)
        self.result_label = ttk.Label(out)
        self.result_label.pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(out, textvariable=self.output_path)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.btn_browse_out = ttk.Button(out, text="…", width=3, command=self.on_pick_output)
        self.btn_browse_out.pack(side=tk.LEFT)
        self.btn_open = ttk.Button(out, command=self.on_open_result, state=tk.DISABLED)
        self.btn_open.pack(side=tk.LEFT, padx=(8, 0))
        self.btn_copy = ttk.Button(out, command=self.on_copy_result, state=tk.DISABLED)
        self.btn_copy.pack(side=tk.LEFT, padx=(8, 0))

        progress = ttk.Frame(self.root_frame)
        progress.pack(fill=tk.X, pady=(8, 0))
        self.progress_label = ttk.Label(
            progress,
            textvariable=self.progress_text,
            width=36,
            style="TLabel",
        )
        self.progress_label.pack(side=tk.RIGHT)
        self.progress_bar = ttk.Progressbar(
            progress,
            mode="determinate",
            variable=self.progress_value,
            maximum=1,
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.log_frame = ttk.LabelFrame(self.root_frame, padding=4)
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = scrolledtext.ScrolledText(self.log_frame, height=12, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state=tk.DISABLED)

    def apply_language(self) -> None:
        self.title(t("gui.title"))
        self.privacy_label.configure(text=t("gui.privacy"))
        self.theme_label.configure(text=t("gui.theme"))
        self.engine_label.configure(text=t("gui.engine"))
        self.layout_label.configure(text=t("gui.layout"))
        self.lang_label.configure(text=t("gui.ui_lang"))
        self.btn_install.configure(text=t("gui.install"))
        self.btn_models.configure(text=t("gui.models"))
        self.btn_tesseract.configure(text=t("gui.tesseract_install"))
        self.btn_refresh.configure(text=t("gui.refresh"))
        self.drop_frame.configure(text=t("gui.document"))
        self.drop_label.configure(text=t("gui.drop_hint"))
        self.btn_file.configure(text=t("gui.pick_file"))
        self.btn_folder.configure(text=t("gui.pick_folder"))
        self.spell_check.configure(text=t("gui.spell"))
        self.fix_caps_check.configure(text=t("gui.fix_caps"))
        self.strip_junk_check.configure(text=t("gui.strip_junk"))
        self.invert_check.configure(text=t("gui.invert"))
        self.result_label.configure(text=t("gui.result"))
        self.btn_open.configure(text=t("gui.open"))
        self.btn_copy.configure(text=t("gui.copy"))
        self.log_frame.configure(text=t("gui.log"))
        if not self._busy:
            self.progress_text.set(t("gui.progress"))
        self.refresh_status()

    def apply_theme(self) -> None:
        name = normalize_theme(self.ui_theme.get())
        self.ui_theme.set(name)
        palette = THEMES[name]
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.configure(bg=palette["bg"])
        style.configure(
            ".",
            background=palette["bg"],
            foreground=palette["fg"],
            fieldbackground=palette["field"],
            troughcolor=palette["trough"],
            bordercolor=palette["bg"],
            darkcolor=palette["bg"],
            lightcolor=palette["bg"],
        )
        style.configure("TFrame", background=palette["bg"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["fg"])
        style.configure("TLabelframe", background=palette["bg"], foreground=palette["fg"])
        style.configure(
            "TLabelframe.Label",
            background=palette["bg"],
            foreground=palette["fg"],
        )
        style.configure(
            "TButton",
            background=palette["button"],
            foreground=palette["fg"],
        )
        style.map(
            "TButton",
            background=[
                ("active", palette["button_active"]),
                ("disabled", palette["button_disabled"]),
            ],
            # Keep label readable on gray disabled buttons (was same as bg).
            foreground=[("disabled", palette["button_disabled_fg"])],
        )
        style.configure(
            "StatusOk.TLabel",
            background=palette["bg"],
            foreground=palette["ok"],
        )
        style.configure(
            "StatusBad.TLabel",
            background=palette["bg"],
            foreground=palette["bad"],
        )
        style.configure(
            "TCheckbutton",
            background=palette["bg"],
            foreground=palette["fg"],
        )
        style.map(
            "TCheckbutton",
            background=[("active", palette["bg"])],
            foreground=[("disabled", palette["fg"])],
        )
        style.configure(
            "TEntry",
            fieldbackground=palette["field"],
            foreground=palette["fg"],
            insertcolor=palette["fg"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=palette["field"],
            foreground=palette["fg"],
            background=palette["field"],
            arrowcolor=palette["fg"],
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["field"])],
            foreground=[("readonly", palette["fg"])],
            background=[("readonly", palette["field"])],
        )
        style.configure(
            "TProgressbar",
            background=palette["accent"],
            troughcolor=palette["trough"],
        )
        style.configure(
            "Wait.TLabel",
            background=palette["bg"],
            foreground=palette["wait"],
        )
        self.option_add("*TCombobox*Listbox.background", palette["field"])
        self.option_add("*TCombobox*Listbox.foreground", palette["fg"])
        self.option_add("*TCombobox*Listbox.selectBackground", palette["select"])
        self.option_add("*TCombobox*Listbox.selectForeground", palette["fg"])
        self.log.configure(
            background=palette["field"],
            foreground=palette["fg"],
            insertbackground=palette["fg"],
            selectbackground=palette["select"],
            selectforeground=palette["fg"],
        )

    def on_language_change(self, _event: object = None) -> None:
        lang = set_ui_lang(self.ui_lang.get())
        self.ui_lang.set(lang)
        save_ui_pref(lang)
        self.apply_language()

    def on_theme_change(self, _event: object = None) -> None:
        theme = normalize_theme(self.ui_theme.get())
        self.ui_theme.set(theme)
        save_theme_pref(theme)
        self.apply_theme()

    def on_layout_change(self, _event: object = None) -> None:
        layout = normalize_layout(self.ui_layout.get())
        self.ui_layout.set(layout)
        save_layout_pref(layout)

    def on_engine_change(self, _event: object = None) -> None:
        engine = normalize_engine(self.ui_engine.get())
        self.ui_engine.set(engine)
        save_engine_pref(engine)
        if self._engine_ready and self._engine.engine_name == engine:
            return
        if self._busy:
            return
        if engine == ENGINE_TESSERACT and not is_tesseract_ready():
            if self._confirm_tesseract_install():
                self._start_tesseract_install()
                return
            fallback = self._last_good_engine or ENGINE_EASY
            self.ui_engine.set(fallback)
            save_engine_pref(fallback)
            return
        # Abort in-flight start so combo is never stuck on _engine_starting.
        self._engine.stop()
        self._engine_start_gen += 1
        self._engine_starting = False
        self._engine_ready = False
        self.append_log(t("gui.engine_restart", engine=engine))
        self._maybe_start_engine(force=True)

    def _setup_drag_and_drop(self) -> None:
        try:
            import windnd
        except ImportError:
            self.append_log(t("gui.dnd_after_install"))
            return

        def on_drop(files: list) -> None:
            # Must not touch Tk widgets here (runs inside hooked WndProc).
            try:
                paths = [decode_drop_path(item) for item in files]
                self._drop_queue.put(paths)
            except Exception as error:  # noqa: BLE001
                self._drop_queue.put(error)

        # Unicode API — required for Cyrillic paths on Windows.
        windnd.hook_dropfiles(self, func=on_drop, force_unicode=True)
        self.append_log(t("gui.dnd_ready"))

    def _poll_drop_queue(self) -> None:
        try:
            while True:
                item = self._drop_queue.get_nowait()
                if isinstance(item, BaseException):
                    messagebox.showerror("OCR", str(item))
                    self.append_log(t("gui.error", title="drag-and-drop", error=item))
                else:
                    self.process_paths(item)
        except queue.Empty:
            pass
        self.after(200, self._poll_drop_queue)

    def _poll_engine_ui_queue(self) -> None:
        try:
            while True:
                item = self._engine_ui_queue.get_nowait()
                kind = item[0]
                gen = item[1]
                if gen != self._engine_start_gen:
                    continue
                if kind == "progress":
                    self._on_engine_progress(item[2], item[3])
                elif kind == "ready":
                    self._on_engine_ready()
                elif kind == "failed":
                    self._on_engine_failed(item[2])
        except queue.Empty:
            pass
        self.after(50, self._poll_engine_ui_queue)

    def _paint_status_placeholder(self) -> None:
        self.status_line.set("…")
        self.status_bar.configure(style="StatusBad.TLabel")
        self.btn_file.configure(state=tk.DISABLED)
        self.btn_folder.configure(state=tk.DISABLED)
        self.progress_text.set(t("gui.progress"))

    def _boot(self) -> None:
        self.base_python = discover_base_python()
        self.refresh_status()
        self._maybe_start_engine()

    def refresh_status(self) -> None:
        python_ok = self.base_python is not None
        libs_ok = libs_installed()
        models_ok = models_ready()
        tess_ok = is_tesseract_ready()
        engine_name = normalize_engine(self.ui_engine.get())
        backend_ready = models_ok if engine_name == ENGINE_EASY else tess_ok

        missing: list[str] = []
        if not python_ok:
            missing.append(t("gui.status_item_python"))
        if not libs_ok:
            missing.append(t("gui.status_item_libs"))
        if engine_name == ENGINE_EASY and not models_ok:
            missing.append(t("gui.status_item_models"))
        if engine_name == ENGINE_TESSERACT and not tess_ok:
            missing.append(t("gui.status_item_tesseract"))
        if libs_ok and backend_ready and not self._engine_ready and not self._engine_starting:
            missing.append(t("gui.status_item_engine"))

        if not missing and self._engine_ready:
            self.status_line.set(t("gui.status_ok_line"))
            self.status_bar.configure(style="StatusOk.TLabel")
        elif self._engine_starting and not missing:
            self.status_line.set(t("gui.status_starting_line"))
            self.status_bar.configure(style="StatusOk.TLabel")
        else:
            detail = ", ".join(missing) if missing else t("gui.status_item_engine")
            self.status_line.set(t("gui.status_bad_line", items=detail))
            self.status_bar.configure(style="StatusBad.TLabel")

        can_run = (
            libs_ok
            and backend_ready
            and self._engine_ready
            and not self._busy
            and not self._engine_starting
        )
        state = tk.NORMAL if can_run else tk.DISABLED
        self.btn_file.configure(state=state)
        self.btn_folder.configure(state=state)
        # Already installed → gray / not clickable (same as busy).
        self.btn_install.configure(
            state=tk.DISABLED if self._busy or libs_ok else tk.NORMAL
        )
        self.btn_models.configure(
            state=tk.DISABLED if self._busy or not libs_ok or models_ok else tk.NORMAL
        )
        self.btn_tesseract.configure(
            state=tk.DISABLED if self._busy or tess_ok else tk.NORMAL
        )
        self.btn_refresh.configure(state=tk.DISABLED if self._busy else tk.NORMAL)

    def _maybe_start_engine(self, *, force: bool = False) -> None:
        if not force and (self._engine_ready or self._engine_starting):
            return
        if self._engine_starting:
            return
        if not libs_installed():
            return
        engine_name = normalize_engine(self.ui_engine.get())
        if engine_name == ENGINE_EASY and not models_ready():
            return
        if engine_name == ENGINE_TESSERACT and not is_tesseract_ready():
            return

        self._engine_start_gen += 1
        start_gen = self._engine_start_gen
        self._engine_starting = True
        self._engine_ready = False
        self.refresh_status()
        self.append_log(t("gui.loading_models"))
        self._set_progress(0, 3)

        def target() -> None:
            try:
                self._engine.start(
                    engine=engine_name,
                    on_progress=lambda current, total: self._engine_ui_queue.put(
                        ("progress", start_gen, current, total)
                    ),
                )
            except Exception as error:  # noqa: BLE001
                self._engine_ui_queue.put(("failed", start_gen, error))
            else:
                self._engine_ui_queue.put(("ready", start_gen))

        threading.Thread(target=target, daemon=True).start()

    def _on_engine_progress(self, current: int, total: int) -> None:
        self._set_progress(current, total)
        self.progress_text.set(t("gui.progress_file", current=current, total=total))

    def _on_engine_ready(self) -> None:
        self._engine_starting = False
        self._engine_ready = True
        self._last_good_engine = normalize_engine(self.ui_engine.get())
        self._set_progress(3, 3)
        self.append_log(t("gui.engine_ready"))
        self.refresh_status()

    def _on_engine_failed(self, error: Exception) -> None:
        failed = normalize_engine(self.ui_engine.get())
        self._engine_starting = False
        self._engine_ready = False
        self._reset_progress()
        self.append_log(t("gui.engine_failed", error=error))
        self.refresh_status()
        messagebox.showerror("OCR", t("gui.engine_failed", error=error))
        # Restore last working engine so UI is not stuck without a worker.
        fallback = self._last_good_engine or ENGINE_EASY
        if failed != fallback and libs_installed():
            if fallback == ENGINE_EASY and not models_ready():
                return
            self.ui_engine.set(fallback)
            save_engine_pref(fallback)
            self.append_log(t("gui.engine_restart", engine=fallback))
            self._maybe_start_engine(force=True)

    def _restart_engine(self) -> None:
        self._engine.stop()
        self._engine_ready = False
        self._engine_starting = False
        self._maybe_start_engine()

    def on_close(self) -> None:
        self._engine.stop()
        self.destroy()

    def on_install(self) -> None:
        self._run_background("gui.task_install", self._install_libraries)

    def on_download_models(self) -> None:
        self._run_background("gui.task_models", self._download_models)

    def on_install_tesseract(self) -> None:
        if not self._confirm_tesseract_install():
            return
        self._start_tesseract_install()

    def _confirm_tesseract_install(self) -> bool:
        """Modal trust dialog: clickable release link + copy SHA-256 before UAC."""
        return ask_tesseract_trust(self)

    def _start_tesseract_install(self) -> None:
        self._run_background("gui.task_tesseract", self._install_tesseract)

    def _install_tesseract(self) -> None:
        install_tesseract_local(self._log_threadsafe)

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

    def on_copy_result(self) -> None:
        path = self._last_result
        if path is None or not path.exists():
            candidate = Path(self.output_path.get().strip() or DEFAULT_OUTPUT)
            path = candidate if candidate.exists() else None
        if path is None:
            messagebox.showinfo("OCR", t("gui.copy_empty"))
            return
        text = path.read_text(encoding="utf-8")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.append_log(t("gui.copied"))

    def on_paste_image(self, _event: object = None) -> str:
        """Recognize an image currently on the Windows clipboard (Win+Shift+S)."""
        if self._busy or self._engine_starting:
            return "break"
        path = save_clipboard_image(APP_DIR / ".clipboard_paste.png")
        if path is None:
            messagebox.showinfo("OCR", t("gui.paste_no_image"))
            return "break"
        self.append_log(t("gui.paste_ok", path=path))
        self.process_paths([path])
        return "break"

    def process_paths(self, paths: Sequence[Path]) -> None:
        if self._busy or self._engine_starting:
            return
        if not self._engine_ready:
            messagebox.showinfo("OCR", t("gui.engine_not_ready"))
            return
        try:
            images = collect_images(paths, recursive=False)
        except ImagePathError as error:
            messagebox.showwarning("OCR", t(error.key, **error.kwargs))
            return
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
        if title_key == "gui.task_ocr":
            self._stop_wait_timer()
            self._set_progress(1, 1)
        title = t(title_key)
        self.append_log(t("gui.done", title=title))
        self.refresh_status()
        if title_key == "gui.task_install":
            messagebox.showinfo("OCR", t("gui.install_done_body"))
            self._restart_engine()
        elif title_key == "gui.task_models":
            self._restart_engine()
        elif title_key == "gui.task_tesseract":
            messagebox.showinfo("OCR", t("gui.tesseract_done_body"))
            if normalize_engine(self.ui_engine.get()) == ENGINE_TESSERACT:
                self._restart_engine()

    def _on_worker_error(self, title_key: str, error: Exception) -> None:
        self._busy = False
        if title_key in {"gui.task_ocr", "gui.task_models", "gui.task_tesseract"}:
            self._stop_wait_timer()
            self._reset_progress()
        title = t(title_key)
        self.append_log(t("gui.error", title=title, error=error))
        self.refresh_status()
        messagebox.showerror("OCR", f"{title}\n\n{error}")

    def _install_libraries(self) -> None:
        if self.base_python is None:
            raise RuntimeError(t("gui.python_required"))
        invalidate_libs_ready_cache()
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

        invalidate_libs_ready_cache()
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
        if not self._engine_ready:
            raise RuntimeError(t("gui.engine_not_ready"))

        total = len(images)
        self.after(0, lambda: self._start_wait_timer("gui.wait_ocr"))
        self._log_threadsafe(t("gui.files_count", count=total))

        def on_file(current: int, max_count: int, path: str) -> None:
            self._log_threadsafe(path)
            detail = t("gui.progress_file", current=current, total=max_count)
            self.after(0, lambda text=detail: self._set_wait_detail(text))
            self.after(0, self._reset_row_progress)

        def on_rows(count: int) -> None:
            self.after(0, lambda n=count: self._begin_row_progress(n))

        def on_line(text: str) -> None:
            if text:
                self._log_threadsafe(text)
            self.after(0, self._advance_row_progress)

        self._engine.recognize(
            images,
            output,
            ui=get_ui_lang(),
            spell=bool(self.spell_enabled.get()),
            fix_caps=bool(self.fix_caps_enabled.get()),
            strip_junk=bool(self.strip_junk_enabled.get()),
            invert=bool(self.invert_enabled.get()),
            layout=normalize_layout(self.ui_layout.get()),
            on_file=on_file,
            on_line=on_line,
            on_rows=on_rows,
        )
        self._last_result = output
        self.after(0, lambda: self.btn_open.configure(state=tk.NORMAL))
        self.after(0, lambda: self.btn_copy.configure(state=tk.NORMAL))
        self._log_threadsafe(t("gui.saved", path=output))

    def _start_wait_timer(self, message_key: str) -> None:
        """Red wait label + elapsed clock; determinate bar stays calm (no pulse)."""
        self._stop_wait_timer()
        self._wait_animating = True
        self._wait_message_key = message_key
        self._wait_detail = ""
        self._wait_started_at = time.monotonic()
        self._row_done = 0
        self._row_total = 0
        self.progress_label.configure(style="Wait.TLabel")
        self.progress_bar.configure(mode="determinate", maximum=1)
        self.progress_value.set(0)
        self._tick_wait_timer()

    def _set_wait_detail(self, detail: str) -> None:
        self._wait_detail = detail

    def _reset_row_progress(self) -> None:
        self._row_done = 0
        self._row_total = 0
        self.progress_bar.configure(mode="determinate", maximum=1)
        self.progress_value.set(0)

    def _begin_row_progress(self, total: int) -> None:
        self._row_total = max(0, int(total))
        self._row_done = 0
        maximum = max(1, self._row_total)
        self.progress_bar.configure(mode="determinate", maximum=maximum)
        self.progress_value.set(0)

    def _advance_row_progress(self) -> None:
        if self._row_total <= 0:
            return
        self._row_done = min(self._row_total, self._row_done + 1)
        self.progress_value.set(self._row_done)

    def _tick_wait_timer(self) -> None:
        if not self._wait_animating:
            return
        elapsed = int(max(0.0, time.monotonic() - self._wait_started_at))
        minutes, seconds = divmod(elapsed, 60)
        clock = f"{minutes}:{seconds:02d}"
        base = t(self._wait_message_key)
        if self._row_total > 0:
            rows = t(
                "gui.progress_file",
                current=self._row_done,
                total=self._row_total,
            )
            self.progress_text.set(f"{base} {rows} · {clock}")
        elif self._wait_detail:
            self.progress_text.set(f"{base} {self._wait_detail} · {clock}")
        else:
            self.progress_text.set(f"{base} · {clock}")
        self.after(1000, self._tick_wait_timer)

    def _stop_wait_timer(self) -> None:
        self._wait_animating = False
        self.progress_label.configure(style="TLabel")
        self.progress_bar.configure(mode="determinate")

    def _set_progress(self, value: float, maximum: int) -> None:
        self._stop_wait_timer()
        maximum = max(1, int(maximum))
        self.progress_bar.configure(mode="determinate", maximum=maximum)
        self.progress_value.set(value)
        self.progress_text.set(
            t("gui.progress_file", current=int(value), total=maximum)
        )

    def _reset_progress(self) -> None:
        self._stop_wait_timer()
        self.progress_bar.configure(mode="determinate", maximum=1)
        self.progress_value.set(0)
        self.progress_text.set(t("gui.progress"))

    def _log_threadsafe(self, message: str) -> None:
        self.after(0, lambda: self.append_log(message))

    def append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


def decode_drop_path(item: object) -> Path:
    """Convert a windnd drop entry (str or bytes) into a Path."""
    if isinstance(item, bytes):
        for encoding in (sys.getfilesystemencoding(), "utf-8", "mbcs", "cp1251"):
            if not encoding:
                continue
            try:
                return Path(item.decode(encoding))
            except UnicodeDecodeError:
                continue
        return Path(item.decode("utf-8", errors="replace"))
    return Path(str(item))


def ask_tesseract_trust(parent: tk.Tk) -> bool:
    """Show official source, clickable release link, and copyable SHA-256 before UAC."""
    info = tesseract_trust_info()
    result = {"ok": False}

    dialog = tk.Toplevel(parent)
    dialog.title(t("gui.tesseract_trust_title"))
    dialog.transient(parent)
    dialog.resizable(False, False)
    dialog.grab_set()

    frame = ttk.Frame(dialog, padding=14)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text=t("gui.tesseract_trust_intro"), wraplength=480, justify=tk.LEFT).pack(
        anchor=tk.W
    )
    ttk.Label(frame, text=t("gui.tesseract_trust_publisher", publisher=info["publisher"])).pack(
        anchor=tk.W, pady=(10, 0)
    )
    ttk.Label(frame, text=t("gui.tesseract_trust_version", version=info["version"])).pack(
        anchor=tk.W
    )

    ttk.Label(frame, text=t("gui.tesseract_trust_release_label")).pack(anchor=tk.W, pady=(10, 0))
    release_link = ttk.Label(
        frame,
        text=info["release_page"],
        foreground="#0b57d0",
        cursor="hand2",
        wraplength=480,
    )
    release_link.pack(anchor=tk.W)
    release_link.bind(
        "<Button-1>",
        lambda _event: webbrowser.open(info["release_page"]),
    )

    ttk.Label(frame, text=t("gui.tesseract_trust_url_label")).pack(anchor=tk.W, pady=(10, 0))
    url_link = ttk.Label(
        frame,
        text=info["url"],
        foreground="#0b57d0",
        cursor="hand2",
        wraplength=480,
    )
    url_link.pack(anchor=tk.W)
    url_link.bind("<Button-1>", lambda _event: webbrowser.open(info["url"]))

    ttk.Label(frame, text=t("gui.tesseract_trust_hash_label")).pack(anchor=tk.W, pady=(10, 0))
    hash_url = f"https://www.virustotal.com/gui/file/{info['sha256']}"
    hash_link = ttk.Label(
        frame,
        text=info["sha256"],
        foreground="#0b57d0",
        cursor="hand2",
        wraplength=480,
    )
    hash_link.pack(anchor=tk.W)
    hash_link.bind("<Button-1>", lambda _event: webbrowser.open(hash_url))
    ttk.Label(
        frame,
        text=t("gui.tesseract_trust_hash_hint"),
        wraplength=480,
    ).pack(anchor=tk.W, pady=(2, 0))

    hash_row = ttk.Frame(frame)
    hash_row.pack(fill=tk.X, pady=(6, 0))

    def copy_hash() -> None:
        dialog.clipboard_clear()
        dialog.clipboard_append(info["sha256"])
        dialog.update()
        copy_btn.configure(text=t("gui.tesseract_trust_copied"))

    copy_btn = ttk.Button(hash_row, text=t("gui.tesseract_trust_copy"), command=copy_hash)
    copy_btn.pack(side=tk.LEFT)
    ttk.Button(
        hash_row,
        text=t("gui.tesseract_trust_open_vt"),
        command=lambda: webbrowser.open(hash_url),
    ).pack(side=tk.LEFT, padx=(8, 0))

    ttk.Label(
        frame,
        text=t("gui.tesseract_trust_warning"),
        wraplength=480,
        justify=tk.LEFT,
    ).pack(anchor=tk.W, pady=(12, 0))

    buttons = ttk.Frame(frame)
    buttons.pack(fill=tk.X, pady=(14, 0))

    def cancel() -> None:
        result["ok"] = False
        dialog.destroy()

    def accept() -> None:
        result["ok"] = True
        dialog.destroy()

    ttk.Button(buttons, text=t("gui.tesseract_trust_cancel"), command=cancel).pack(side=tk.RIGHT)
    ttk.Button(buttons, text=t("gui.tesseract_trust_continue"), command=accept).pack(
        side=tk.RIGHT, padx=(0, 8)
    )

    dialog.protocol("WM_DELETE_WINDOW", cancel)
    dialog.update_idletasks()
    dialog.geometry(
        f"+{parent.winfo_rootx() + 40}+{parent.winfo_rooty() + 40}"
    )
    parent.wait_window(dialog)
    return bool(result["ok"])


def save_clipboard_image(dest: Path) -> Optional[Path]:
    """Persist a clipboard bitmap (or return an existing clipboard image path)."""
    content = ImageGrab.grabclipboard()
    if content is None:
        return None
    if isinstance(content, list):
        for item in content:
            path = Path(str(item))
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
            }:
                return path
        return None
    if isinstance(content, Image.Image):
        image = content.convert("RGB") if content.mode != "RGB" else content
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="PNG")
        return dest
    return None


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


def normalize_theme(value: str) -> str:
    name = (value or "").strip().lower()
    return name if name in THEMES else DEFAULT_THEME


def load_theme_pref() -> str:
    try:
        if THEME_PREF_FILE.exists():
            return normalize_theme(THEME_PREF_FILE.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    return DEFAULT_THEME


def save_theme_pref(theme: str) -> None:
    try:
        THEME_PREF_FILE.write_text(normalize_theme(theme) + "\n", encoding="utf-8")
    except OSError:
        return


def load_engine_pref() -> str:
    try:
        if ENGINE_PREF_FILE.exists():
            return normalize_engine(ENGINE_PREF_FILE.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    except Exception:  # noqa: BLE001 — bad pref falls back to default
        pass
    return DEFAULT_ENGINE


def save_engine_pref(engine: str) -> None:
    try:
        ENGINE_PREF_FILE.write_text(normalize_engine(engine) + "\n", encoding="utf-8")
    except OSError:
        return


def load_layout_pref() -> str:
    try:
        if LAYOUT_PREF_FILE.exists():
            return normalize_layout(LAYOUT_PREF_FILE.read_text(encoding="utf-8").strip())
    except OSError:
        pass
    except Exception:  # noqa: BLE001 — bad pref falls back to default
        pass
    return DEFAULT_LAYOUT


def save_layout_pref(layout: str) -> None:
    try:
        LAYOUT_PREF_FILE.write_text(normalize_layout(layout) + "\n", encoding="utf-8")
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
            **hidden_subprocess_kwargs(),
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
            **hidden_subprocess_kwargs(),
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
            **hidden_subprocess_kwargs(),
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


def libs_installed() -> bool:
    """Fast check: venv exists and easyocr package files are present (no import)."""
    if _LIBS_READY_CACHE is True:
        return True
    if not VENV_PYTHON.exists():
        return False
    site = VENV_DIR / "Lib" / "site-packages"
    return site.is_dir() and any(site.glob("easyocr*"))


def libs_ready() -> bool:
    """Import check used after install. Cached after first success."""
    global _LIBS_READY_CACHE
    if _LIBS_READY_CACHE is True:
        return True
    if not VENV_PYTHON.exists():
        _LIBS_READY_CACHE = False
        return False
    try:
        completed = subprocess.run(
            [str(VENV_PYTHON), "-c", "import easyocr, cv2, PIL"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            **hidden_subprocess_kwargs(),
        )
    except (OSError, subprocess.TimeoutExpired):
        _LIBS_READY_CACHE = False
        return False
    ready = completed.returncode == 0
    _LIBS_READY_CACHE = ready
    return ready


def invalidate_libs_ready_cache() -> None:
    global _LIBS_READY_CACHE
    _LIBS_READY_CACHE = None


def models_ready() -> bool:
    return MODELS_DIR.is_dir() and any(MODELS_DIR.glob("*.pth"))


def hidden_subprocess_kwargs() -> dict:
    """Prevent black console flashes on Windows GUI subprocesses."""
    if os.name != "nt":
        return {}
    # Available on Windows since Python 3.7.
    flag = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return {"creationflags": flag} if flag else {}


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
        **hidden_subprocess_kwargs(),
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
