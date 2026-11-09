"""Локальное распознавание текста с фотографии документа A4.

Путь одного запуска:

1. Разобрать аргументы командной строки.
2. Либо один раз скачать модели (--download-models), либо распознать фото.
3. Проверить, что файл существует и формат поддерживается.
4. Прочитать изображение (в том числе путь с кириллицей).
5. Подготовить фото, если не передан --no-preprocess.
6. Распознать текст локальной моделью EasyOCR на CPU.
7. Напечатать текст и записать его в result.txt.

Распознавание всегда создаёт Reader с download_enabled=False.
Скачивание возможно только в ветке --download-models.
Подготовка кадра вынесена в preprocess.py — там нет сети и OCR.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from preprocess import find_document_corners, preprocess

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_LANGUAGES = "ru+en"
DEFAULT_OUTPUT = Path("result.txt")
MODEL_DIR = Path(__file__).resolve().parent / "models"
LINE_GROUP_PIXELS = 15

# EasyOCR кладёт в models detection и recognition. Без .pth файлов
# офлайн-распознавание даже не начинаем — так ошибка яснее, чем у библиотеки.
MODEL_FILE_GLOB = "*.pth"


class OcrError(Exception):
    """Ошибка, которую можно показать пользователю без трассировки."""


class TextLine(NamedTuple):
    text: str
    confidence: float
    box: tuple[tuple[int, int], ...]


class RecognizedDocument(NamedTuple):
    path: Path
    lines: tuple[TextLine, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Читает аргументы. Сам по себе сеть не использует и файлы не открывает."""
    parser = argparse.ArgumentParser(
        description="Распознаёт текст с фотографии документа A4. Работает локально, без сети.",
        epilog=(
            "Примеры:\n"
            "  python ocr.py document.jpg\n"
            "  python ocr.py document.jpg --lang ru+en\n"
            "  python ocr.py document.jpg -o result.txt\n"
            "  python ocr.py page1.jpg page2.png\n"
            "  python ocr.py --download-models\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Пути к изображениям: .jpg, .jpeg, .png, .webp, .bmp.",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANGUAGES,
        help="Языки через + или запятую. По умолчанию: ru+en.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Файл для текста. По умолчанию: result.txt.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Дополнительно сохранить координаты блоков и уверенность распознавания.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Отдать в OCR исходное фото, без выравнивания и контраста.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Один раз скачать модели в папку models. Единственная команда, которой нужен интернет.",
    )
    return parser.parse_args(argv)


def parse_languages(value: str) -> list[str]:
    """Превращает 'ru+en' или 'ru,en' в список кодов без повторов."""
    languages: list[str] = []
    for part in value.replace("+", ",").split(","):
        code = part.strip().lower()
        if code and code not in languages:
            languages.append(code)
    if not languages:
        raise OcrError("Укажите хотя бы один язык, например --lang ru+en.")
    return languages


def run(args: argparse.Namespace) -> None:
    """Выполняет одну из двух команд: загрузка моделей или распознавание."""
    languages = parse_languages(args.lang)

    if args.download_models:
        if args.images:
            raise OcrError(
                "Загрузка моделей запускается отдельно, без путей к фото:\n"
                "  python ocr.py --download-models"
            )
        download_models(languages)
        return

    if not args.images:
        raise OcrError(
            "Укажите путь к фотографии документа.\n"
            "Пример: python ocr.py document.jpg"
        )

    for image_path in args.images:
        ensure_supported_image(image_path)

    reader = create_reader(languages, allow_download=False)
    documents: list[RecognizedDocument] = []
    for image_path in args.images:
        documents.append(
            recognize_file(
                reader,
                image_path,
                preprocess_image=not args.no_preprocess,
            )
        )

    text = render_documents(documents)
    save_text(args.output, text)
    print_result(text, args.output)

    if args.debug:
        debug_path = debug_output_path(args.output)
        save_text(debug_path, render_debug(documents))
        print(f"Подробности сохранены в {debug_path}")


def download_models(languages: Sequence[str]) -> None:
    """Единственный сценарий, в котором EasyOCR может открыть сеть."""
    print("Скачивание моделей EasyOCR. Интернет нужен только для этого шага.")
    print("Языки: " + ", ".join(languages))
    create_reader(languages, allow_download=True)
    print(f"Модели сохранены в {MODEL_DIR}")
    print("Дальше распознавание работает без интернета.")


def ensure_supported_image(path: Path) -> None:
    """Проверяет путь до чтения пикселей и до запуска OCR."""
    if not path.exists():
        raise OcrError(f"Файл не найден: {path}")
    if not path.is_file():
        raise OcrError(f"Это не файл: {path}")

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return

    allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    extension = suffix if suffix else "(без расширения)"
    raise OcrError(f"Формат {extension} не поддерживается. Допустимо: {allowed}.")


def recognize_file(reader: Any, path: Path, *, preprocess_image: bool) -> RecognizedDocument:
    """Читает один файл и возвращает строки в порядке чтения."""
    image = load_image(path)
    try:
        prepared = preprocess(image) if preprocess_image else image
    except cv2.error as error:
        raise OcrError(f"Не удалось обработать изображение {path.name}: {error}") from error
    lines = read_lines(reader, prepared, path)
    return RecognizedDocument(path=path, lines=tuple(lines))


def load_image(path: Path) -> np.ndarray:
    """Читает изображение в формате BGR, который ждёт OpenCV.

    cv2.imread на Windows теряет пути с кириллицей, поэтому байты читает numpy.
    """
    file_bytes = np.fromfile(str(path), dtype=np.uint8)
    if file_bytes.size == 0:
        raise OcrError(f"Файл пустой: {path}")

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    return load_image_with_pillow(path)


def load_image_with_pillow(path: Path) -> np.ndarray:
    """Запасной способ чтения. На выходе тоже BGR, как у OpenCV."""
    try:
        with Image.open(path) as pil_image:
            rgb = np.array(pil_image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise OcrError(f"Не удалось открыть изображение: {path}") from error

    if rgb.size == 0:
        raise OcrError(f"Не удалось открыть изображение: {path}")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def local_models_present() -> bool:
    """True, если в models уже лежат веса EasyOCR."""
    return MODEL_DIR.is_dir() and any(MODEL_DIR.glob(MODEL_FILE_GLOB))


def create_reader(languages: Sequence[str], *, allow_download: bool) -> Any:
    """Создаёт локальный OCR.

    allow_download=True допустим только из download_models.
    Распознавание вызывает эту функцию с allow_download=False.
    Перед офлайн-запуском проверяем папку models сами: иначе EasyOCR
    может вести себя менее предсказуемо, чем наша ошибка.
    """
    easyocr = import_easyocr()
    if allow_download:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    elif not local_models_present():
        raise missing_models_error(allow_download=False)

    try:
        with silence_library_noise():
            return easyocr.Reader(
                list(languages),
                gpu=False,
                model_storage_directory=str(MODEL_DIR),
                download_enabled=allow_download,
                verbose=allow_download,
            )
    except FileNotFoundError as error:
        raise missing_models_error(allow_download) from error
    except ValueError as error:
        raise OcrError(f"EasyOCR не принял языки {list(languages)}: {error}") from error
    except OSError as error:
        if allow_download:
            raise OcrError(f"Не удалось скачать модели: {error}") from error
        raise missing_models_error(allow_download) from error


def import_easyocr() -> Any:
    try:
        import easyocr
    except ModuleNotFoundError as error:
        raise OcrError(
            "Не установлен пакет easyocr. Установите зависимости из README, раздел «Установка»."
        ) from error
    return easyocr


def missing_models_error(allow_download: bool) -> OcrError:
    if allow_download:
        return OcrError(
            "Не удалось получить модели EasyOCR. Проверьте интернет и повторите:\n"
            "  python ocr.py --download-models"
        )
    return OcrError(
        "Локальные модели не найдены в папке models.\n"
        "Один раз выполните: python ocr.py --download-models\n"
        "Для этой команды нужен интернет. Само распознавание в сеть не обращается."
    )


@contextmanager
def silence_library_noise() -> Iterator[None]:
    """Скрывает предупреждения torch/EasyOCR. Наш вывод для пользователя не меняет."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def read_lines(reader: Any, image: np.ndarray, path: Path) -> list[TextLine]:
    """Запускает OCR и сортирует блоки сверху вниз, слева направо."""
    try:
        with silence_library_noise():
            raw_rows = reader.readtext(image, detail=1, paragraph=False)
    except Exception as error:
        raise OcrError(f"Не удалось распознать файл {path.name}: {error}") from error

    lines: list[TextLine] = []
    for box, text, confidence in raw_rows:
        points = tuple((int(x), int(y)) for x, y in box)
        lines.append(TextLine(text=str(text), confidence=float(confidence), box=points))
    return sorted(lines, key=reading_order_key)


def reading_order_key(line: TextLine) -> tuple[int, int]:
    """Ключ порядка чтения. Близкие по вертикали блоки попадают в одну строку."""
    top = min(y for _x, y in line.box)
    left = min(x for x, _y in line.box)
    row = top // LINE_GROUP_PIXELS
    return row, left


def render_documents(documents: Sequence[RecognizedDocument]) -> str:
    """Один файл — только текст. Несколько файлов — текст с заголовком имени."""
    if len(documents) == 1:
        return document_text(documents[0])

    blocks = [
        f"=== {document.path.name} ===\n{document_text(document)}"
        for document in documents
    ]
    return "\n\n".join(blocks)


def document_text(document: RecognizedDocument) -> str:
    parts = [line.text.strip() for line in document.lines if line.text.strip()]
    if not parts:
        return "Текст не распознан."
    return "\n".join(parts)


def render_debug(documents: Sequence[RecognizedDocument]) -> str:
    """Текст для человека: номер строки, уверенность и четыре угла блока."""
    blocks: list[str] = []
    for document in documents:
        rows = [f"Файл: {document.path}"]
        if not document.lines:
            rows.append("Текст не распознан.")
        for number, line in enumerate(document.lines, start=1):
            rows.append(f"{number}. [{line.confidence:.2f}] {line.text}")
            rows.append(f"    координаты: {format_box(line.box)}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def format_box(box: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"({x}, {y})" for x, y in box)


def debug_output_path(output: Path) -> Path:
    """result.txt -> result.debug.txt, рядом с основным результатом."""
    suffix = output.suffix or ".txt"
    return output.with_name(f"{output.stem}.debug{suffix}")


def save_text(path: Path, text: str) -> None:
    """Пишет UTF-8 и завершающий перевод строки. Существующий файл заменяется."""
    directory = path.parent
    if str(directory) not in ("", ".") and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(text)
        if not text.endswith("\n"):
            output.write("\n")


def print_result(text: str, output: Path) -> None:
    print("Распознанный текст:")
    print(text)
    print(f"Текст сохранён в {output}")


def configure_console() -> None:
    """Просит консоль Windows принять UTF-8. Если это не вышло, работу не прерывает."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            return


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_console()
    try:
        run(parse_args(argv))
    except OcrError as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
