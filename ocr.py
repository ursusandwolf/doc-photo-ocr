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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

# EasyOCR подключается в create_reader. Там же зафиксирован офлайн-режим,
# поэтому поиск по download_enabled показывает единственные две ветки.

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_LANGUAGES = "ru+en"
DEFAULT_OUTPUT = Path("result.txt")
MODEL_DIR = Path(__file__).resolve().parent / "models"

# Лёгкая предобработка. Жёсткую бинаризацию не делаем:
# EasyOCR на фото читает серое изображение лучше, чем чёрно-белое.
MEDIAN_BLUR_SIZE = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# Контур листа ищем на уменьшенной копии: полный кадр телефона здесь не нужен.
DETECT_MAX_SIDE = 1000
MIN_DOCUMENT_AREA_RATIO = 0.30
MAX_DOCUMENT_AREA_RATIO = 0.95
APPROX_EPSILON_RATIO = 0.02
MIN_SHEET_SIDE = 100
MIN_SIDE_TO_LONG_SIDE = 0.20
BORDER_MARGIN_RATIO = 0.02

# Выравниваем только небольшой уверенный наклон. Большой угол чаще означает
# ошибку оценки, чем реально повёрнутый лист.
MIN_SKEW_DEGREES = 0.4
MAX_SKEW_DEGREES = 15.0
MIN_TEXT_PIXELS = 100

# Слова одной строки имеют близкую верхнюю границу. Группируем их перед сортировкой.
LINE_GROUP_PIXELS = 15


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

    # Сначала проверяем все пути. Иначе при ошибке во втором файле
    # первый уже был бы распознан, а result.txt ещё не записан.
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
    Если OpenCV не знает кодек, тот же файл пробует открыть Pillow.
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


def preprocess(image: np.ndarray) -> np.ndarray:
    """Готовит фото к OCR и возвращает одноканальное изображение.

    Сначала геометрия: лист на столе мог быть снят с перспективой и наклоном.
    Потом лёгкий шум и локальный контраст. Бинаризации нет намеренно.
    """
    aligned = straighten_document(image)
    gray = to_gray(aligned)
    denoised = cv2.medianBlur(gray, MEDIAN_BLUR_SIZE)
    contrasted = increase_contrast(denoised)
    return deskew(contrasted)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def increase_contrast(gray: np.ndarray) -> np.ndarray:
    """Поднимает локальный контраст. На крошечном кадре возвращает его как есть.

    CLAHE делит картинку на плитки. Если кадр меньше плитки, делить нечего.
    """
    height, width = gray.shape[:2]
    tile_width, tile_height = CLAHE_TILE_SIZE
    if height < tile_height or width < tile_width:
        return gray

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    return clahe.apply(gray)


def straighten_document(image: np.ndarray) -> np.ndarray:
    """Выравнивает перспективу, только если контур похож на отдельный лист.

    Скан, который уже занимает весь кадр, возвращается без изменений.
    Ошибочный контур тоже не применяется: лучше оставить исходное фото.
    """
    corners = find_document_corners(image)
    if corners is None:
        return image
    return warp_document(image, corners)


def find_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Ищет четыре угла листа. None — выравнивать перспективу не нужно."""
    small, scale = resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    largest = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    image_area = float(small.shape[0] * small.shape[1])

    for contour in largest:
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, APPROX_EPSILON_RATIO * perimeter, closed=True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        area_ratio = cv2.contourArea(approx) / image_area
        if not MIN_DOCUMENT_AREA_RATIO <= area_ratio <= MAX_DOCUMENT_AREA_RATIO:
            continue

        points = approx.reshape(4, 2).astype(np.float32) / scale
        ordered = order_corners(points)
        if accept_document_corners(ordered, image.shape):
            return ordered
    return None


def resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Уменьшает кадр для поиска контура. scale переводит точки обратно в исходный размер."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= DETECT_MAX_SIDE:
        return image, 1.0

    scale = DETECT_MAX_SIDE / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def order_corners(points: np.ndarray) -> np.ndarray:
    """Порядок углов: левый верх, правый верх, правый низ, левый низ.

    У левого верхнего угла сумма x + y наименьшая, у правого нижнего — наибольшая.
    Разность y - x разделяет правый верхний и левый нижний углы.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    difference = points[:, 1] - points[:, 0]
    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]
    return ordered


def accept_document_corners(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """Отсекает слишком узкие контуры и контур, совпадающий с самим кадром."""
    sides = corner_side_lengths(corners)
    longest = max(sides)
    if longest < MIN_SHEET_SIDE:
        return False
    if min(sides) < MIN_SIDE_TO_LONG_SIDE * longest:
        return False
    return corners_are_inside_frame(corners, image_shape)


def corner_side_lengths(corners: np.ndarray) -> list[float]:
    pairs = (
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    )
    return [float(np.linalg.norm(end - start)) for start, end in pairs]


def corners_are_inside_frame(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """Лист на фотографии стола не прижат всеми углами к границе кадра.

    Если все четыре угла на границе, это сам кадр (типичный скан), а не лист.
    Такой кадр перспективой не трогаем.
    """
    height, width = image_shape[:2]
    margin_x = width * BORDER_MARGIN_RATIO
    margin_y = height * BORDER_MARGIN_RATIO
    corners_on_border = 0

    for x, y in corners:
        on_border = (
            x <= margin_x
            or y <= margin_y
            or x >= width - 1 - margin_x
            or y >= height - 1 - margin_y
        )
        if on_border:
            corners_on_border += 1

    return corners_on_border < 4


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Переносит найденный четырёхугольник в прямоугольный вид листа."""
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    if width < MIN_SHEET_SIDE or height < MIN_SHEET_SIDE:
        return image

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def deskew(gray: np.ndarray) -> np.ndarray:
    """Поворачивает текст к горизонтали. Неуверенный угол оставляет как есть."""
    angle = estimate_skew_degrees(gray)
    if angle is None:
        return gray

    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_skew_degrees(gray: np.ndarray) -> Optional[float]:
    """Оценивает наклон по тёмным пикселям текста. None — поворот применять нельзя."""
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    points = cv2.findNonZero(binary)
    if points is None or len(points) < MIN_TEXT_PIXELS:
        return None

    _center, _size, raw_angle = cv2.minAreaRect(points)
    angle = rotation_to_level_text(float(raw_angle))
    if abs(angle) < MIN_SKEW_DEGREES or abs(angle) > MAX_SKEW_DEGREES:
        return None
    return angle


def rotation_to_level_text(raw_angle: float) -> float:
    """Переводит угол minAreaRect в угол для cv2.getRotationMatrix2D.

    OpenCV возвращает угол стороны прямоугольника в диапазоне [-90, 0).
    Для горизонтального текста он близок к 0 или к -90. Ниже — обычное
    приведение этого значения к малому повороту изображения.
    """
    if raw_angle < -45.0:
        return -(90.0 + raw_angle)
    return -raw_angle


def create_reader(languages: Sequence[str], *, allow_download: bool) -> Any:
    """Создаёт локальный OCR.

    allow_download=True допустим только из download_models.
    Распознавание вызывает эту функцию с allow_download=False и при отсутствии
    моделей получает ошибку, а не попытку выйти в сеть.
    gpu=False зафиксирован: программа должна одинаково работать без видеокарты.
    """
    easyocr = import_easyocr()
    if allow_download:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
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


def read_lines(reader: Any, image: np.ndarray, path: Path) -> list[TextLine]:
    """Запускает OCR и сортирует блоки сверху вниз, слева направо."""
    try:
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
