"""Простой GUI для локального OCR документа A4.

1. Находит Python 3.9+ (системный, иначе runtime из zip).
2. Ставит библиотеки в локальный .venv только если их ещё нет.
3. Качает модели только если их ещё нет.
4. Принимает файл/папку и вызывает ocr.py через .venv.
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

APP_DIR = Path(__file__).resolve().parent
VENV_DIR = APP_DIR / ".venv"
VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
REQUIREMENTS = APP_DIR / "requirements.txt"
MODELS_DIR = APP_DIR / "models"
OCR_SCRIPT = APP_DIR / "ocr.py"
DEFAULT_OUTPUT = APP_DIR / "result.txt"
RUNTIME_PYTHON = APP_DIR / "runtime" / "python.exe"
MIN_PYTHON = (3, 9)
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("OCR документа A4")
        self.geometry("720x560")
        self.minsize(640, 480)

        self.base_python = discover_base_python()
        self.output_path = tk.StringVar(value=str(DEFAULT_OUTPUT))
        self.status_python = tk.StringVar()
        self.status_libs = tk.StringVar()
        self.status_models = tk.StringVar()
        self._busy = False
        self._last_result: Optional[Path] = None

        self._build()
        self.refresh_status()
        self._setup_drag_and_drop()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            root,
            text="Локальное распознавание. Данные в сеть не отправляются.",
        ).pack(anchor=tk.W)

        status = ttk.LabelFrame(root, text="Состояние", padding=8)
        status.pack(fill=tk.X, pady=(10, 8))
        ttk.Label(status, textvariable=self.status_python).pack(anchor=tk.W)
        ttk.Label(status, textvariable=self.status_libs).pack(anchor=tk.W)
        ttk.Label(status, textvariable=self.status_models).pack(anchor=tk.W)

        actions = ttk.Frame(root)
        actions.pack(fill=tk.X, pady=4)
        self.btn_install = ttk.Button(
            actions, text="Установить библиотеки", command=self.on_install
        )
        self.btn_install.pack(side=tk.LEFT)
        self.btn_models = ttk.Button(
            actions, text="Загрузить модели", command=self.on_download_models
        )
        self.btn_models.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Обновить статус", command=self.refresh_status).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        drop = ttk.LabelFrame(root, text="Документ", padding=8)
        drop.pack(fill=tk.X, pady=8)
        self.drop_label = ttk.Label(
            drop,
            text="Перетащите сюда фото или папку\nили выберите кнопкой ниже",
            anchor=tk.CENTER,
            justify=tk.CENTER,
            padding=18,
        )
        self.drop_label.pack(fill=tk.X)

        buttons = ttk.Frame(drop)
        buttons.pack(fill=tk.X, pady=(8, 0))
        self.btn_file = ttk.Button(buttons, text="Выбрать файл…", command=self.on_pick_file)
        self.btn_file.pack(side=tk.LEFT)
        self.btn_folder = ttk.Button(
            buttons, text="Выбрать папку…", command=self.on_pick_folder
        )
        self.btn_folder.pack(side=tk.LEFT, padx=(8, 0))

        out = ttk.Frame(root)
        out.pack(fill=tk.X, pady=4)
        ttk.Label(out, text="Результат:").pack(side=tk.LEFT)
        ttk.Entry(out, textvariable=self.output_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=6
        )
        ttk.Button(out, text="…", width=3, command=self.on_pick_output).pack(side=tk.LEFT)
        self.btn_open = ttk.Button(
            out, text="Открыть", command=self.on_open_result, state=tk.DISABLED
        )
        self.btn_open.pack(side=tk.LEFT, padx=(8, 0))

        log_frame = ttk.LabelFrame(root, text="Журнал", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.configure(state=tk.DISABLED)

    def _setup_drag_and_drop(self) -> None:
        try:
            import windnd
        except ImportError:
            self.append_log(
                "Drag-and-drop появится после «Установить библиотеки» (пакет windnd)."
            )
            return

        def on_drop(files: list[bytes]) -> None:
            paths = [Path(item.decode("utf-8", errors="replace")) for item in files]
            self.after(0, lambda: self.process_paths(paths))

        windnd.hook_dropfiles(self, func=on_drop)
        self.append_log("Drag-and-drop включён: перетащите файл или папку в окно.")

    def refresh_status(self) -> None:
        if self.base_python is None:
            self.status_python.set("Python: не найден (нужен 3.9+ с tkinter или папка runtime)")
        else:
            version = python_version_string(self.base_python)
            source = "системный" if is_system_python(self.base_python) else "runtime из zip"
            self.status_python.set(f"Python: {version} ({source})")

        if libs_ready():
            self.status_libs.set("Библиотеки: готовы (.venv)")
        elif VENV_DIR.exists():
            self.status_libs.set("Библиотеки: .venv есть, но EasyOCR ещё не установлен")
        else:
            self.status_libs.set("Библиотеки: не установлены")

        if models_ready():
            self.status_models.set("Модели: готовы (models)")
        else:
            self.status_models.set("Модели: не загружены")

        can_run = libs_ready() and models_ready() and not self._busy
        state = tk.NORMAL if can_run else tk.DISABLED
        self.btn_file.configure(state=state)
        self.btn_folder.configure(state=state)
        self.btn_install.configure(state=tk.DISABLED if self._busy else tk.NORMAL)
        self.btn_models.configure(
            state=tk.DISABLED if self._busy or not libs_ready() else tk.NORMAL
        )

    def on_install(self) -> None:
        self._run_background("Установка библиотек", self._install_libraries)

    def on_download_models(self) -> None:
        self._run_background("Загрузка моделей", self._download_models)

    def on_pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите фото документа",
            filetypes=[
                ("Изображения", "*.jpg *.jpeg *.png *.webp *.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self.process_paths([Path(path)])

    def on_pick_folder(self) -> None:
        path = filedialog.askdirectory(title="Выберите папку с фото")
        if path:
            self.process_paths([Path(path)])

    def on_pick_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Куда сохранить текст",
            defaultextension=".txt",
            initialfile="result.txt",
            filetypes=[("Текст", "*.txt")],
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
            messagebox.showwarning(
                "OCR",
                "Не найдено изображений .jpg/.jpeg/.png/.webp/.bmp",
            )
            return
        output = Path(self.output_path.get().strip() or DEFAULT_OUTPUT)
        self._run_background("Распознавание", lambda: self._recognize(images, output))

    def _run_background(self, title: str, worker: Callable[[], None]) -> None:
        if self._busy:
            return
        self._busy = True
        self.refresh_status()
        self.append_log(f"--- {title} ---")

        def target() -> None:
            try:
                worker()
            except Exception as error:  # noqa: BLE001
                self.after(0, lambda: self._on_worker_error(title, error))
            else:
                self.after(0, lambda: self._on_worker_done(title))

        threading.Thread(target=target, daemon=True).start()

    def _on_worker_done(self, title: str) -> None:
        self._busy = False
        self.append_log(f"Готово: {title}")
        self.refresh_status()
        if title == "Установка библиотек":
            messagebox.showinfo(
                "OCR",
                "Библиотеки установлены.\n"
                "Перезапустите программу, чтобы включить drag-and-drop,\n"
                "затем нажмите «Загрузить модели».",
            )

    def _on_worker_error(self, title: str, error: Exception) -> None:
        self._busy = False
        self.append_log(f"Ошибка ({title}): {error}")
        self.refresh_status()
        messagebox.showerror("OCR", f"{title}\n\n{error}")

    def _install_libraries(self) -> None:
        if self.base_python is None:
            raise RuntimeError(
                "Не найден Python 3.9+ с tkinter. "
                "Установите Python с python.org или используйте релиз с папкой runtime."
            )
        if libs_ready():
            self._log_threadsafe("Библиотеки уже установлены — повторный pip не нужен.")
            return

        if not VENV_DIR.exists():
            self._log_threadsafe(f"Создаю .venv через {self.base_python}")
            run_checked(
                [str(self.base_python), "-m", "venv", str(VENV_DIR)],
                self._log_threadsafe,
            )

        py = str(VENV_PYTHON)
        self._log_threadsafe("Обновляю pip…")
        run_checked([py, "-m", "pip", "install", "--upgrade", "pip"], self._log_threadsafe)

        self._log_threadsafe("Ставлю PyTorch (CPU)…")
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

        self._log_threadsafe("Ставлю зависимости из requirements.txt…")
        run_checked(
            [py, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            self._log_threadsafe,
        )

        if not libs_ready():
            raise RuntimeError("Установка завершилась, но import easyocr не удался.")
        self._log_threadsafe("Библиотеки готовы.")

    def _download_models(self) -> None:
        if not libs_ready():
            raise RuntimeError("Сначала установите библиотеки.")
        if models_ready():
            self._log_threadsafe("Модели уже есть — загрузка не нужна.")
            return

        self._log_threadsafe("Скачиваю модели EasyOCR (нужен интернет)…")
        run_checked(
            [str(VENV_PYTHON), str(OCR_SCRIPT), "--download-models"],
            self._log_threadsafe,
            cwd=APP_DIR,
        )
        if not models_ready():
            raise RuntimeError("Модели не появились в папке models.")
        self._log_threadsafe("Модели готовы. Дальше программа работает офлайн.")

    def _recognize(self, images: Sequence[Path], output: Path) -> None:
        if not libs_ready():
            raise RuntimeError("Сначала установите библиотеки.")
        if not models_ready():
            raise RuntimeError("Сначала загрузите модели.")

        command = [
            str(VENV_PYTHON),
            str(OCR_SCRIPT),
            "-o",
            str(output),
            *[str(path) for path in images],
        ]
        self._log_threadsafe("Файлов: " + str(len(images)))
        run_checked(command, self._log_threadsafe, cwd=APP_DIR)
        self._last_result = output
        self.after(0, lambda: self.btn_open.configure(state=tk.NORMAL))
        self._log_threadsafe(f"Текст сохранён: {output}")

    def _log_threadsafe(self, message: str) -> None:
        self.after(0, lambda: self.append_log(message))

    def append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


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
        raise RuntimeError(f"Команда завершилась с кодом {code}")


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
