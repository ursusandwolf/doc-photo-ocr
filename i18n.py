"""User-facing UI strings. Default language is Russian; English is optional.

Code comments and README stay in English. Only runtime help / GUI / errors go through here.
"""

from __future__ import annotations

from typing import Any

DEFAULT_UI = "ru"
SUPPORTED_UI = ("ru", "en")

_ui_lang = DEFAULT_UI

# Keys are stable English identifiers. Values are locale strings.
_MESSAGES: dict[str, dict[str, str]] = {
    "ru": {
        "cli.description": "Распознаёт текст с фотографии документа A4. Работает локально, без сети.",
        "cli.epilog": (
            "Примеры:\n"
            "  python ocr.py document.jpg\n"
            "  python ocr.py document.jpg --lang ru+en\n"
            "  python ocr.py document.jpg -o result.txt\n"
            "  python ocr.py page1.jpg page2.png\n"
            "  python ocr.py resources/\n"
            "  python ocr.py resources/ --per-file\n"
            "  python ocr.py scans/ --recursive -o batch.txt\n"
            "  python ocr.py --ui en document.jpg\n"
            "  python ocr.py --download-models\n"
        ),
        "cli.images_help": "Файлы и/или папки с изображениями: .jpg, .jpeg, .png, .webp, .bmp.",
        "cli.lang_help": "Языки OCR через + или запятую. По умолчанию: ru.",
        "cli.output_help": "Общий файл результата. По умолчанию: result.txt. Игнорируется с --per-file.",
        "cli.per_file_help": "Пакетный режим: рядом с каждым фото писать отдельный .txt с тем же именем.",
        "cli.recursive_help": "В папках искать изображения во всех вложенных каталогах.",
        "cli.debug_help": "Дополнительно сохранить координаты блоков и уверенность распознавания.",
        "cli.no_preprocess_help": "Отдать в OCR исходное фото, без выравнивания и контраста.",
        "cli.no_spell_help": "Не исправлять орфографию по офлайн-словарю (по умолчанию исправление включено).",
        "cli.download_help": "Один раз скачать модели в папку models. Единственная команда, которой нужен интернет.",
        "cli.worker_help": "Держать EasyOCR в памяти и принимать JSON-задания из stdin (для GUI).",
        "cli.ui_help": "Язык интерфейса: ru (по умолчанию) или en.",
        "err.need_ocr_lang": "Укажите хотя бы один язык OCR, например --lang ru.",
        "err.download_with_images": (
            "Загрузка моделей запускается отдельно, без путей к фото:\n"
            "  python ocr.py --download-models"
        ),
        "err.worker_with_images": (
            "Режим --worker запускается без путей к фото:\n"
            "  python ocr.py --worker"
        ),
        "err.need_input": (
            "Укажите путь к фотографии или папке.\n"
            "Пример: python ocr.py document.jpg\n"
            "Пакетно: python ocr.py resources/ --per-file"
        ),
        "err.no_images": "Не найдено изображений .jpg/.jpeg/.png/.webp/.bmp по указанным путям.",
        "err.path_missing": "Путь не найден: {path}",
        "err.bad_format": "Формат {ext} не поддерживается. Допустимо: {allowed}.",
        "err.not_file_or_dir": "Это не файл и не папка: {path}",
        "err.no_docs": "Нет документов для сохранения.",
        "err.file_missing": "Файл не найден: {path}",
        "err.not_file": "Это не файл: {path}",
        "err.empty_file": "Файл пустой: {path}",
        "err.open_image": "Не удалось открыть изображение: {path}",
        "err.preprocess": "Не удалось обработать изображение {name}: {error}",
        "err.recognize": "Не удалось распознать файл {name}: {error}",
        "err.easyocr_langs": "EasyOCR не принял языки {langs}: {error}",
        "err.download_failed": "Не удалось скачать модели: {error}",
        "err.no_easyocr": (
            "Не установлен пакет easyocr. Установите зависимости из README, раздел Installation."
        ),
        "err.models_download_failed": (
            "Не удалось получить модели EasyOCR. Проверьте интернет и повторите:\n"
            "  python ocr.py --download-models"
        ),
        "err.models_missing": (
            "Локальные модели не найдены в папке models.\n"
            "Один раз выполните: python ocr.py --download-models\n"
            "Для этой команды нужен интернет. Само распознавание в сеть не обращается."
        ),
        "msg.files_count": "Файлов к распознаванию: {count}",
        "msg.processing": "Обработка: {path}",
        "msg.debug_saved": "Подробности сохранены в {path}",
        "msg.batch_saved": "Пакетно сохранено файлов: {count}",
        "msg.download_start": "Скачивание моделей EasyOCR. Интернет нужен только для этого шага.",
        "msg.languages": "Языки: {langs}",
        "msg.models_saved": "Модели сохранены в {path}",
        "msg.offline_ready": "Дальше распознавание работает без интернета.",
        "msg.recognized_header": "Распознанный текст:",
        "msg.text_saved": "Текст сохранён в {path}",
        "msg.error_prefix": "Ошибка: {error}",
        "msg.no_text": "Текст не распознан.",
        "msg.debug_file": "Файл: {path}",
        "msg.debug_coords": "    координаты: {box}",
        "msg.no_extension": "(без расширения)",
        "gui.title": "OCR документа A4",
        "gui.privacy": "Локальное распознавание. Данные в сеть не отправляются.",
        "gui.status": "Состояние",
        "gui.install": "Установить библиотеки",
        "gui.models": "Загрузить модели",
        "gui.refresh": "Обновить статус",
        "gui.document": "Документ",
        "gui.drop_hint": "Перетащите сюда фото или папку\nили выберите кнопкой ниже",
        "gui.pick_file": "Выбрать файл…",
        "gui.pick_folder": "Выбрать папку…",
        "gui.result": "Результат:",
        "gui.spell": "Исправлять орфографию",
        "gui.open": "Открыть",
        "gui.log": "Журнал",
        "gui.ui_lang": "Язык:",
        "gui.python_missing": "Python: не найден (нужен 3.9+ с tkinter или папка runtime)",
        "gui.python_ok": "Python: {version} ({source})",
        "gui.source_system": "системный",
        "gui.source_runtime": "runtime из zip",
        "gui.libs_ready": "Библиотеки: готовы (.venv)",
        "gui.libs_partial": "Библиотеки: .venv есть, но EasyOCR ещё не установлен",
        "gui.libs_missing": "Библиотеки: не установлены",
        "gui.models_ready": "Модели: готовы (models)",
        "gui.models_missing": "Модели: не загружены",
        "gui.dnd_after_install": "Drag-and-drop появится после «Установить библиотеки» (пакет windnd).",
        "gui.dnd_ready": "Drag-and-drop включён: перетащите файл или папку в окно.",
        "gui.pick_file_title": "Выберите фото документа",
        "gui.pick_folder_title": "Выберите папку с фото",
        "gui.pick_output_title": "Куда сохранить текст",
        "gui.filetype_images": "Изображения",
        "gui.filetype_all": "Все файлы",
        "gui.filetype_text": "Текст",
        "gui.no_images": "Не найдено изображений .jpg/.jpeg/.png/.webp/.bmp",
        "gui.task_install": "Установка библиотек",
        "gui.task_models": "Загрузка моделей",
        "gui.task_ocr": "Распознавание",
        "gui.done": "Готово: {title}",
        "gui.error": "Ошибка ({title}): {error}",
        "gui.install_done_body": (
            "Библиотеки установлены.\n"
            "Перезапустите программу, чтобы включить drag-and-drop,\n"
            "затем нажмите «Загрузить модели»."
        ),
        "gui.python_required": (
            "Не найден Python 3.9+ с tkinter. "
            "Установите Python с python.org или используйте релиз с папкой runtime."
        ),
        "gui.libs_already": "Библиотеки уже установлены — повторный pip не нужен.",
        "gui.creating_venv": "Создаю .venv через {python}",
        "gui.upgrade_pip": "Обновляю pip…",
        "gui.install_torch": "Ставлю PyTorch (CPU)…",
        "gui.install_reqs": "Ставлю зависимости из requirements.txt…",
        "gui.libs_import_failed": "Установка завершилась, но import easyocr не удался.",
        "gui.libs_ready_log": "Библиотеки готовы.",
        "gui.need_libs_first": "Сначала установите библиотеки.",
        "gui.models_already": "Модели уже есть — загрузка не нужна.",
        "gui.models_downloading": "Скачиваю модели EasyOCR (нужен интернет)…",
        "gui.models_missing_after": "Модели не появились в папке models.",
        "gui.models_ready_log": "Модели готовы. Дальше программа работает офлайн.",
        "gui.need_models_first": "Сначала загрузите модели.",
        "gui.files_count": "Файлов: {count}",
        "gui.progress": "Прогресс",
        "gui.progress_file": "{current}/{total}",
        "gui.loading_models": "Загрузка моделей…",
        "gui.wait_ocr": "Ждите, идёт распознавание",
        "gui.engine_ready": "Движок OCR готов.",
        "gui.engine_not_ready": "Сначала дождитесь загрузки моделей.",
        "gui.engine_failed": "Не удалось загрузить модели: {error}",
        "gui.saved": "Текст сохранён: {path}",
    },
    "en": {
        "cli.description": "Recognize text from an A4 document photo. Runs locally, offline.",
        "cli.epilog": (
            "Examples:\n"
            "  python ocr.py document.jpg\n"
            "  python ocr.py document.jpg --lang ru+en\n"
            "  python ocr.py document.jpg -o result.txt\n"
            "  python ocr.py page1.jpg page2.png\n"
            "  python ocr.py resources/\n"
            "  python ocr.py resources/ --per-file\n"
            "  python ocr.py scans/ --recursive -o batch.txt\n"
            "  python ocr.py --ui en document.jpg\n"
            "  python ocr.py --download-models\n"
        ),
        "cli.images_help": "Image files and/or folders: .jpg, .jpeg, .png, .webp, .bmp.",
        "cli.lang_help": "OCR languages joined by + or comma. Default: ru.",
        "cli.output_help": "Shared output file. Default: result.txt. Ignored with --per-file.",
        "cli.per_file_help": "Batch mode: write a matching .txt next to each image.",
        "cli.recursive_help": "Search images in nested folders as well.",
        "cli.debug_help": "Also save block coordinates and confidence scores.",
        "cli.no_preprocess_help": "Send the original photo to OCR without geometry/contrast tweaks.",
        "cli.no_spell_help": "Disable offline dictionary spell-fix (enabled by default).",
        "cli.download_help": "Download models into models/ once. The only command that needs the internet.",
        "cli.worker_help": "Keep EasyOCR in memory and accept JSON jobs from stdin (for the GUI).",
        "cli.ui_help": "UI language: ru (default) or en.",
        "err.need_ocr_lang": "Provide at least one OCR language, e.g. --lang ru.",
        "err.download_with_images": (
            "Model download runs alone, without image paths:\n"
            "  python ocr.py --download-models"
        ),
        "err.worker_with_images": (
            "Worker mode runs without image paths:\n"
            "  python ocr.py --worker"
        ),
        "err.need_input": (
            "Provide a photo or folder path.\n"
            "Example: python ocr.py document.jpg\n"
            "Batch: python ocr.py resources/ --per-file"
        ),
        "err.no_images": "No .jpg/.jpeg/.png/.webp/.bmp images found for the given paths.",
        "err.path_missing": "Path not found: {path}",
        "err.bad_format": "Format {ext} is not supported. Allowed: {allowed}.",
        "err.not_file_or_dir": "Not a file or folder: {path}",
        "err.no_docs": "No documents to save.",
        "err.file_missing": "File not found: {path}",
        "err.not_file": "Not a file: {path}",
        "err.empty_file": "File is empty: {path}",
        "err.open_image": "Could not open image: {path}",
        "err.preprocess": "Could not preprocess image {name}: {error}",
        "err.recognize": "Could not recognize file {name}: {error}",
        "err.easyocr_langs": "EasyOCR rejected languages {langs}: {error}",
        "err.download_failed": "Could not download models: {error}",
        "err.no_easyocr": (
            "Package easyocr is not installed. Install dependencies from the README Installation section."
        ),
        "err.models_download_failed": (
            "Could not obtain EasyOCR models. Check the internet and retry:\n"
            "  python ocr.py --download-models"
        ),
        "err.models_missing": (
            "Local models were not found in the models folder.\n"
            "Run once: python ocr.py --download-models\n"
            "That command needs the internet. Recognition itself does not go online."
        ),
        "msg.files_count": "Files to recognize: {count}",
        "msg.processing": "Processing: {path}",
        "msg.debug_saved": "Debug details saved to {path}",
        "msg.batch_saved": "Batch files written: {count}",
        "msg.download_start": "Downloading EasyOCR models. Internet is required only for this step.",
        "msg.languages": "Languages: {langs}",
        "msg.models_saved": "Models saved to {path}",
        "msg.offline_ready": "Recognition can run offline from now on.",
        "msg.recognized_header": "Recognized text:",
        "msg.text_saved": "Text saved to {path}",
        "msg.error_prefix": "Error: {error}",
        "msg.no_text": "No text recognized.",
        "msg.debug_file": "File: {path}",
        "msg.debug_coords": "    coordinates: {box}",
        "msg.no_extension": "(no extension)",
        "gui.title": "A4 Document OCR",
        "gui.privacy": "Local recognition. No data is sent to the network.",
        "gui.status": "Status",
        "gui.install": "Install libraries",
        "gui.models": "Download models",
        "gui.refresh": "Refresh status",
        "gui.document": "Document",
        "gui.drop_hint": "Drop a photo or folder here\nor use the buttons below",
        "gui.pick_file": "Choose file…",
        "gui.pick_folder": "Choose folder…",
        "gui.result": "Output:",
        "gui.spell": "Fix spelling",
        "gui.open": "Open",
        "gui.log": "Log",
        "gui.ui_lang": "Language:",
        "gui.python_missing": "Python: not found (need 3.9+ with tkinter or a runtime folder)",
        "gui.python_ok": "Python: {version} ({source})",
        "gui.source_system": "system",
        "gui.source_runtime": "runtime from zip",
        "gui.libs_ready": "Libraries: ready (.venv)",
        "gui.libs_partial": "Libraries: .venv exists, but EasyOCR is not installed yet",
        "gui.libs_missing": "Libraries: not installed",
        "gui.models_ready": "Models: ready (models)",
        "gui.models_missing": "Models: not downloaded",
        "gui.dnd_after_install": "Drag-and-drop appears after Install libraries (windnd package).",
        "gui.dnd_ready": "Drag-and-drop enabled: drop a file or folder onto the window.",
        "gui.pick_file_title": "Choose a document photo",
        "gui.pick_folder_title": "Choose a folder with photos",
        "gui.pick_output_title": "Where to save the text",
        "gui.filetype_images": "Images",
        "gui.filetype_all": "All files",
        "gui.filetype_text": "Text",
        "gui.no_images": "No .jpg/.jpeg/.png/.webp/.bmp images found",
        "gui.task_install": "Installing libraries",
        "gui.task_models": "Downloading models",
        "gui.task_ocr": "Recognition",
        "gui.done": "Done: {title}",
        "gui.error": "Error ({title}): {error}",
        "gui.install_done_body": (
            "Libraries are installed.\n"
            "Restart the app to enable drag-and-drop,\n"
            "then click Download models."
        ),
        "gui.python_required": (
            "Python 3.9+ with tkinter was not found. "
            "Install Python from python.org or use a release that includes runtime."
        ),
        "gui.libs_already": "Libraries are already installed — pip will not run again.",
        "gui.creating_venv": "Creating .venv with {python}",
        "gui.upgrade_pip": "Upgrading pip…",
        "gui.install_torch": "Installing PyTorch (CPU)…",
        "gui.install_reqs": "Installing requirements.txt…",
        "gui.libs_import_failed": "Install finished, but import easyocr failed.",
        "gui.libs_ready_log": "Libraries are ready.",
        "gui.need_libs_first": "Install libraries first.",
        "gui.models_already": "Models are already present — download is not needed.",
        "gui.models_downloading": "Downloading EasyOCR models (internet required)…",
        "gui.models_missing_after": "Models did not appear in the models folder.",
        "gui.models_ready_log": "Models are ready. The app works offline after this.",
        "gui.need_models_first": "Download models first.",
        "gui.files_count": "Files: {count}",
        "gui.progress": "Progress",
        "gui.progress_file": "{current}/{total}",
        "gui.loading_models": "Loading models…",
        "gui.wait_ocr": "Please wait, recognition in progress",
        "gui.engine_ready": "OCR engine is ready.",
        "gui.engine_not_ready": "Wait until models finish loading.",
        "gui.engine_failed": "Failed to load models: {error}",
        "gui.saved": "Text saved: {path}",
    },
}


def get_ui_lang() -> str:
    return _ui_lang


def set_ui_lang(lang: str) -> str:
    """Set UI language. Unknown values fall back to Russian."""
    global _ui_lang
    normalized = (lang or DEFAULT_UI).strip().lower()
    _ui_lang = normalized if normalized in SUPPORTED_UI else DEFAULT_UI
    return _ui_lang


def t(key: str, **kwargs: Any) -> str:
    """Translate a UI key for the active language."""
    table = _MESSAGES.get(_ui_lang) or _MESSAGES[DEFAULT_UI]
    template = table.get(key) or _MESSAGES[DEFAULT_UI].get(key) or key
    if kwargs:
        return template.format(**kwargs)
    return template
