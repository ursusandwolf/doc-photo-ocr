"""Local OCR for A4 document photos.

Run path:

1. Parse CLI arguments.
2. Either download models once (--download-models) or recognize photos.
3. Check that each path exists and the format is supported.
4. Load the image (including Cyrillic paths on Windows).
5. Preprocess unless --no-preprocess is set.
6. Run the selected OCR engine (EasyOCR default, or Tesseract).
7. Print text and write result.txt (or one .txt per file with --per-file).

Inputs may be files and/or folders. A folder takes first-level images;
--recursive also walks nested directories.

Recognition always builds Reader with download_enabled=False.
Network download is allowed only in the --download-models branch.
Image prep lives in preprocess.py — no network and no OCR there.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from i18n import get_ui_lang, set_ui_lang, t
from image_paths import SUPPORTED_EXTENSIONS
from image_paths import PathError as ImagePathError
from image_paths import collect_images as collect_image_paths
from engines import DEFAULT_ENGINE, create_engine, normalize_engine
from errors import OcrError
from offline_guard import offline_session
from preprocess import find_document_corners, invert_image, preprocess
from spellcheck import correct_text

DEFAULT_LANGUAGES = "ru"
DEFAULT_OUTPUT = Path("result.txt")
MODEL_DIR = Path(__file__).resolve().parent / "models"
# Line clustering: prefer vertical overlap; center distance is a fallback.
MIN_VERTICAL_OVERLAP_RATIO = 0.5
MAX_CENTER_Y_RATIO = 0.35

# EasyOCR stores detection/recognition weights under models/.
# Without .pth files we fail early with a clear offline error.
MODEL_FILE_GLOB = "*.pth"


class TextLine(NamedTuple):
    text: str
    confidence: float
    box: tuple[tuple[int, int], ...]


class RecognizedDocument(NamedTuple):
    path: Path
    lines: tuple[TextLine, ...]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args. Does not open files or use the network."""
    argv_list = list(argv) if argv is not None else None

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--ui", choices=("ru", "en"), default="ru")
    pre_args, remaining = pre.parse_known_args(argv_list)
    set_ui_lang(pre_args.ui)

    parser = argparse.ArgumentParser(
        description=t("cli.description"),
        epilog=t("cli.epilog"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[pre],
    )
    parser.add_argument("images", nargs="*", type=Path, help=t("cli.images_help"))
    parser.add_argument("--lang", default=DEFAULT_LANGUAGES, help=t("cli.lang_help"))
    parser.add_argument(
        "--engine",
        default=DEFAULT_ENGINE,
        help=t("cli.engine_help"),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=t("cli.output_help"),
    )
    parser.add_argument("--per-file", action="store_true", help=t("cli.per_file_help"))
    parser.add_argument("--recursive", action="store_true", help=t("cli.recursive_help"))
    parser.add_argument("--debug", action="store_true", help=t("cli.debug_help"))
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help=t("cli.no_preprocess_help"),
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help=t("cli.invert_help"),
    )
    parser.add_argument(
        "--no-spell",
        action="store_true",
        help=t("cli.no_spell_help"),
    )
    parser.add_argument(
        "--no-fix-caps",
        action="store_true",
        help=t("cli.no_fix_caps_help"),
    )
    parser.add_argument(
        "--strip-junk",
        action="store_true",
        help=t("cli.strip_junk_help"),
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help=t("cli.download_help"),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=t("cli.worker_help"),
    )
    # Re-declare for help text after language is known.
    parser.set_defaults(ui=pre_args.ui)
    for action in parser._actions:
        if action.dest == "ui":
            action.help = t("cli.ui_help")
            break
    return parser.parse_args(remaining)


def parse_languages(value: str) -> list[str]:
    """Turn 'ru+en' or 'ru,en' into a de-duplicated language list."""
    languages: list[str] = []
    for part in value.replace("+", ",").split(","):
        code = part.strip().lower()
        if code and code not in languages:
            languages.append(code)
    if not languages:
        raise OcrError(t("err.need_ocr_lang"))
    return languages


def run(args: argparse.Namespace) -> None:
    """Run either model download, GUI worker, or recognition."""
    set_ui_lang(getattr(args, "ui", get_ui_lang()))
    languages = parse_languages(args.lang)

    if args.download_models:
        if args.images:
            raise OcrError(t("err.download_with_images"))
        download_models(languages)
        return

    engine_name = normalize_engine(getattr(args, "engine", DEFAULT_ENGINE))

    if args.worker:
        if args.images:
            raise OcrError(t("err.worker_with_images"))
        run_worker(languages, engine_name=engine_name)
        return

    if not args.images:
        raise OcrError(t("err.need_input"))

    image_paths = collect_images(args.images, recursive=args.recursive)
    if not image_paths:
        raise OcrError(t("err.no_images"))

    for image_path in image_paths:
        ensure_supported_image(image_path)

    print(t("msg.files_count", count=len(image_paths)))
    engine = create_engine(engine_name, languages, allow_download=False)
    documents: list[RecognizedDocument] = []
    with offline_session():
        engine.warm()
        for image_path in image_paths:
            print(t("msg.processing", path=image_path))
            documents.append(
                recognize_file(
                    engine,
                    image_path,
                    preprocess_image=not args.no_preprocess,
                    invert=bool(args.invert),
                )
            )

    spell = not args.no_spell
    fix_caps = not args.no_fix_caps
    strip_junk = bool(args.strip_junk)
    if args.per_file:
        write_per_file_results(
            documents,
            debug=args.debug,
            spell=spell,
            fix_caps=fix_caps,
            strip_junk=strip_junk,
        )
        return

    text = render_documents(
        documents,
        spell=spell,
        fix_caps=fix_caps,
        strip_junk=strip_junk,
    )
    save_text(args.output, text)
    print_result(text, args.output)

    if args.debug:
        debug_path = debug_output_path(args.output)
        save_text(debug_path, render_debug(documents))
        print(t("msg.debug_saved", path=debug_path))


def collect_images(paths: Sequence[Path], *, recursive: bool = False) -> list[Path]:
    """Collect unique images from files and folders in stable order."""
    try:
        return collect_image_paths(paths, recursive=recursive)
    except ImagePathError as error:
        kwargs = dict(error.kwargs)
        if error.key == "err.bad_format" and kwargs.get("ext") == "none":
            kwargs["ext"] = t("msg.no_extension")
        raise OcrError(t(error.key, **kwargs)) from error


def write_per_file_results(
    documents: Sequence[RecognizedDocument],
    *,
    debug: bool,
    spell: bool = True,
    fix_caps: bool = True,
    strip_junk: bool = False,
) -> None:
    """Write document.png -> document.txt next to each input."""
    if not documents:
        raise OcrError(t("err.no_docs"))

    for document in documents:
        output = document.path.with_suffix(".txt")
        text = document_text(
            document,
            spell=spell,
            fix_caps=fix_caps,
            strip_junk=strip_junk,
        )
        save_text(output, text)
        print(f"{document.path.name} -> {output}")
        if debug:
            debug_path = debug_output_path(output)
            save_text(debug_path, render_debug((document,)))
            print(f"  debug: {debug_path}")

    print(t("msg.batch_saved", count=len(documents)))


def download_models(languages: Sequence[str]) -> None:
    """Only path where EasyOCR may open the network."""
    print(t("msg.download_start"))
    print(t("msg.languages", langs=", ".join(languages)))
    create_reader(languages, allow_download=True)
    print(t("msg.models_saved", path=MODEL_DIR))
    print(t("msg.offline_ready"))


def emit_worker(tag: str, payload: str = "") -> None:
    """Machine-readable line for the GUI worker protocol."""
    # One physical line only — newlines would break the GUI reader.
    safe = " ".join(str(payload).split())
    message = f"##{tag}" if not safe else f"##{tag} {safe}"
    print(message, flush=True)


def run_worker(languages: Sequence[str], *, engine_name: str = DEFAULT_ENGINE) -> None:
    """Keep one OCR backend loaded and accept JSON recognize jobs on stdin."""
    engine_name = normalize_engine(engine_name)
    try:
        emit_worker("PROGRESS", "1/3 importing")
        engine = create_engine(engine_name, languages, allow_download=False)
        emit_worker("PROGRESS", "2/3 loading")
        with offline_session():
            engine.warm()
        emit_worker("PROGRESS", "3/3 ready")
        emit_worker("READY")
    except OcrError as error:
        emit_worker("ERROR", str(error))
        return
    except Exception as error:  # noqa: BLE001
        emit_worker("ERROR", str(error))
        return

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            emit_worker("ERROR", f"bad json: {error}")
            continue

        command = str(request.get("cmd", "")).strip().lower()
        if command in {"quit", "exit"}:
            emit_worker("BYE")
            return
        if command == "ping":
            emit_worker("PONG")
            continue
        if command != "recognize":
            emit_worker("ERROR", f"unknown cmd: {command}")
            continue

        try:
            with offline_session():
                handle_worker_recognize(engine, request)
            emit_worker("OK")
        except OcrError as error:
            emit_worker("ERROR", str(error))
        except Exception as error:  # noqa: BLE001
            emit_worker("ERROR", str(error))


def handle_worker_recognize(engine: Any, request: dict[str, Any]) -> None:
    """Recognize one batch from a worker JSON request and write the output file."""
    ui = str(request.get("ui") or get_ui_lang())
    set_ui_lang(ui)
    images = [Path(item) for item in request.get("images") or []]
    if not images:
        raise OcrError(t("err.need_input"))
    output = Path(request.get("output") or DEFAULT_OUTPUT)
    no_preprocess = bool(request.get("no_preprocess", False))
    invert = bool(request.get("invert", False))
    debug = bool(request.get("debug", False))
    per_file = bool(request.get("per_file", False))
    spell = bool(request.get("spell", True))
    fix_caps = bool(request.get("fix_caps", True))
    strip_junk = bool(request.get("strip_junk", False))

    for image_path in images:
        ensure_supported_image(image_path)

    def emit_line(text: str) -> None:
        # Always normalize quotes; other polish follows flags.
        payload = (
            correct_text(
                text,
                spell=spell,
                fix_caps=fix_caps,
                strip_junk=strip_junk,
            )
            if text
            else text
        )
        emit_worker("LINE", payload)

    def emit_rows(count: int) -> None:
        emit_worker("ROWS", str(count))

    emit_worker("FILES", str(len(images)))
    documents: list[RecognizedDocument] = []
    for index, image_path in enumerate(images, start=1):
        emit_worker("FILE", f"{index}/{len(images)} {image_path}")
        documents.append(
            recognize_file(
                engine,
                image_path,
                preprocess_image=not no_preprocess,
                invert=invert,
                on_row=emit_line,
                on_rows=emit_rows,
            )
        )

    if per_file:
        write_per_file_results(
            documents,
            debug=debug,
            spell=spell,
            fix_caps=fix_caps,
            strip_junk=strip_junk,
        )
        return

    text = render_documents(
        documents,
        spell=spell,
        fix_caps=fix_caps,
        strip_junk=strip_junk,
    )
    save_text(output, text)
    if debug:
        save_text(debug_output_path(output), render_debug(documents))


def ensure_supported_image(path: Path) -> None:
    """Validate path before reading pixels or starting OCR."""
    if not path.exists():
        raise OcrError(t("err.file_missing", path=path))
    if not path.is_file():
        raise OcrError(t("err.not_file", path=path))

    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return

    allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    extension = suffix if suffix else t("msg.no_extension")
    raise OcrError(t("err.bad_format", ext=extension, allowed=allowed))


def recognize_file(
    engine: Any,
    path: Path,
    *,
    preprocess_image: bool,
    invert: bool = False,
    on_row: Optional[Callable[[str], None]] = None,
    on_rows: Optional[Callable[[int], None]] = None,
) -> RecognizedDocument:
    """Read one file and return word boxes in reading order.

    on_rows is called once with the number of visual rows after detection.
    on_row is called after each row is recognized (may be empty).
    `engine` is an OcrBackend (EasyOCR or Tesseract).
    """
    image = load_image(path)
    if invert:
        image = invert_image(image)
    try:
        prepared = preprocess(image) if preprocess_image else image
    except cv2.error as error:
        raise OcrError(t("err.preprocess", name=path.name, error=error)) from error
    lines = engine.recognize(
        prepared,
        path,
        on_row=on_row,
        on_rows=on_rows,
    )
    return RecognizedDocument(path=path, lines=tuple(lines))


def load_image(path: Path) -> np.ndarray:
    """Load an image as BGR for OpenCV.

    cv2.imread drops Cyrillic paths on Windows, so numpy reads the bytes.
    """
    file_bytes = np.fromfile(str(path), dtype=np.uint8)
    if file_bytes.size == 0:
        raise OcrError(t("err.empty_file", path=path))

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    return load_image_with_pillow(path)


def load_image_with_pillow(path: Path) -> np.ndarray:
    """Fallback loader. Output is BGR like OpenCV."""
    try:
        with Image.open(path) as pil_image:
            rgb = np.array(pil_image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as error:
        raise OcrError(t("err.open_image", path=path)) from error

    if rgb.size == 0:
        raise OcrError(t("err.open_image", path=path))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def local_models_present() -> bool:
    """True when EasyOCR weight files already exist under models/."""
    return MODEL_DIR.is_dir() and any(MODEL_DIR.glob(MODEL_FILE_GLOB))


def create_reader(languages: Sequence[str], *, allow_download: bool) -> Any:
    """Create a local OCR reader.

    allow_download=True is only for download_models.
    Recognition always calls this with allow_download=False and checks models/
    first so missing weights become our error instead of a network attempt.
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
        raise OcrError(
            t("err.easyocr_langs", langs=list(languages), error=error)
        ) from error
    except OSError as error:
        if allow_download:
            raise OcrError(t("err.download_failed", error=error)) from error
        raise missing_models_error(allow_download) from error


def import_easyocr() -> Any:
    try:
        import easyocr
    except ModuleNotFoundError as error:
        raise OcrError(t("err.no_easyocr")) from error
    return easyocr


def missing_models_error(allow_download: bool) -> OcrError:
    if allow_download:
        return OcrError(t("err.models_download_failed"))
    return OcrError(t("err.models_missing"))


@contextmanager
def silence_library_noise() -> Iterator[None]:
    """Hide torch/EasyOCR warnings without changing our own output."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def read_lines(
    reader: Any,
    image: np.ndarray,
    path: Path,
    *,
    on_row: Optional[Callable[[str], None]] = None,
    on_rows: Optional[Callable[[int], None]] = None,
) -> list[TextLine]:
    """Detect once, recognize row by row, optionally stream each row via on_row."""
    try:
        with silence_library_noise():
            from easyocr.utils import reformat_input

            img, img_cv_grey = reformat_input(image)
            horizontal_list, free_list = reader.detect(img, reformat=False)
            # EasyOCR returns depth-3 lists; readtext unwraps the first page.
            horizontal_list, free_list = horizontal_list[0], free_list[0]
            placeholders = detection_placeholders(horizontal_list, free_list)
            if not placeholders:
                if on_rows is not None:
                    on_rows(0)
                return []

            rows = group_lines_into_rows(placeholders)
            if on_rows is not None:
                on_rows(len(rows))
            lines: list[TextLine] = []
            for row in rows:
                free_boxes = [box_as_free_list_item(item.box) for item in row]
                raw_rows = reader.recognize(
                    img_cv_grey,
                    horizontal_list=[],
                    free_list=free_boxes,
                    detail=1,
                    paragraph=False,
                    reformat=False,
                )
                row_lines = parse_easyocr_results(raw_rows)
                row_lines = sorted(row_lines, key=box_left)
                lines.extend(row_lines)
                row_text = " ".join(part.text for part in row_lines if part.text).strip()
                if on_row is not None:
                    on_row(row_text)
            return lines
    except OcrError:
        raise
    except Exception as error:
        raise OcrError(t("err.recognize", name=path.name, error=error)) from error


def detection_placeholders(
    horizontal_list: Sequence[Any],
    free_list: Sequence[Any],
) -> list[TextLine]:
    """Turn EasyOCR detect boxes into TextLine placeholders for row clustering."""
    placeholders: list[TextLine] = []
    for item in horizontal_list or []:
        box = horizontal_box_to_points(item)
        if box is not None:
            placeholders.append(TextLine(text="", confidence=0.0, box=box))
    for item in free_list or []:
        box = free_box_to_points(item)
        if box is not None:
            placeholders.append(TextLine(text="", confidence=0.0, box=box))
    return placeholders


def horizontal_box_to_points(item: Any) -> Optional[tuple[tuple[int, int], ...]]:
    """Convert [x_min, x_max, y_min, y_max] into a 4-point box."""
    try:
        x_min, x_max, y_min, y_max = (int(value) for value in item[:4])
    except (TypeError, ValueError):
        return None
    return (
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    )


def free_box_to_points(item: Any) -> Optional[tuple[tuple[int, int], ...]]:
    """Convert a free-form 4-point EasyOCR box into int tuples."""
    try:
        points = tuple((int(x), int(y)) for x, y in item)
    except (TypeError, ValueError):
        return None
    if len(points) < 4:
        return None
    return points[:4]


def box_as_free_list_item(box: Sequence[tuple[int, int]]) -> list[list[int]]:
    """EasyOCR free_list entry: [[x,y], ...]."""
    return [[int(x), int(y)] for x, y in box]


def parse_easyocr_results(raw_rows: Sequence[Any]) -> list[TextLine]:
    """Convert EasyOCR detail=1 rows into TextLine values."""
    lines: list[TextLine] = []
    for box, text, confidence in raw_rows:
        points = free_box_to_points(box)
        cleaned = str(text).strip()
        if points is None or not cleaned:
            continue
        lines.append(TextLine(text=cleaned, confidence=float(confidence), box=points))
    return lines


def box_left(line: TextLine) -> int:
    return min(x for x, _y in line.box)


def box_top(line: TextLine) -> int:
    return min(y for _x, y in line.box)


def box_bottom(line: TextLine) -> int:
    return max(y for _x, y in line.box)


def box_center_y(line: TextLine) -> float:
    return (box_top(line) + box_bottom(line)) / 2.0


def box_height(line: TextLine) -> int:
    return max(1, box_bottom(line) - box_top(line))


def box_width(line: TextLine) -> int:
    xs = [x for x, _y in line.box]
    return max(1, max(xs) - min(xs))


def box_reading_angle_degrees(line: TextLine) -> float:
    """Tilt of the longer left-to-right edge (degrees, CCW from +X)."""
    points = list(line.box)
    if len(points) < 2:
        return 0.0
    best_dx = 0.0
    best_dy = 0.0
    best_len = -1.0
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        # Normalize to left -> right so atan2 stays near 0, not ±180.
        if dx < 0:
            dx = -dx
            dy = -dy
        length = math.hypot(dx, dy)
        # Prefer the wider edge; ignore near-vertical sides.
        if length > best_len and abs(dx) >= abs(dy):
            best_len = length
            best_dx = dx
            best_dy = dy
    if best_len <= 0:
        return 0.0
    return math.degrees(math.atan2(best_dy, best_dx))


def estimate_page_skew_degrees(lines: Sequence[TextLine]) -> float:
    """Median text tilt from wide boxes. Positive means text rises to the right."""
    samples: list[tuple[float, float]] = []
    for item in lines:
        width = float(box_width(item))
        height = float(box_height(item))
        if width < 1.5 * height:
            continue
        angle = box_reading_angle_degrees(item)
        samples.append((angle, width))
    if not samples:
        samples = [(box_reading_angle_degrees(item), float(box_width(item))) for item in lines]
    if not samples:
        return 0.0

    # Axis-aligned detector noise (angle≈0) must not cancel a real page skew.
    tilted = [(angle, weight) for angle, weight in samples if abs(angle) >= 2.0]
    chosen = tilted if len(tilted) >= max(2, len(samples) // 3) else samples
    chosen.sort(key=lambda item: item[0])
    return float(chosen[len(chosen) // 2][0])


def rotate_point(
    x: float,
    y: float,
    *,
    angle_degrees: float,
    origin: tuple[float, float],
) -> tuple[float, float]:
    rad = math.radians(angle_degrees)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx = x - origin[0]
    dy = y - origin[1]
    return origin[0] + dx * cos_a - dy * sin_a, origin[1] + dx * sin_a + dy * cos_a


class FlatBox(NamedTuple):
    line: TextLine
    left: float
    top: float
    bottom: float
    center_y: float
    height: float


def flatten_box(line: TextLine, *, skew_degrees: float, origin: tuple[float, float]) -> FlatBox:
    """Rotate box into page-aligned coordinates (negative skew undoes tilt)."""
    rotated = [
        rotate_point(float(x), float(y), angle_degrees=-skew_degrees, origin=origin)
        for x, y in line.box
    ]
    ys = [point[1] for point in rotated]
    xs = [point[0] for point in rotated]
    top = min(ys)
    bottom = max(ys)
    return FlatBox(
        line=line,
        left=min(xs),
        top=top,
        bottom=bottom,
        center_y=(top + bottom) / 2.0,
        height=max(1.0, bottom - top),
    )


def vertical_overlap_ratio(a: FlatBox, b: FlatBox) -> float:
    """overlap_height / min(heights). 1.0 means fully nested on the Y axis."""
    top = max(a.top, b.top)
    bottom = min(a.bottom, b.bottom)
    overlap = max(0.0, bottom - top)
    return overlap / float(min(a.height, b.height))


def row_median_height(row: Sequence[FlatBox]) -> float:
    heights = sorted(item.height for item in row)
    mid = len(heights) // 2
    if not heights:
        return 1.0
    if len(heights) % 2:
        return float(heights[mid])
    return (heights[mid - 1] + heights[mid]) / 2.0


def row_mean_center_y(row: Sequence[FlatBox]) -> float:
    return sum(item.center_y for item in row) / float(len(row))


def fits_text_row(row: Sequence[FlatBox], candidate: FlatBox) -> bool:
    """True when candidate belongs on this visual text line."""
    best_overlap = max(vertical_overlap_ratio(item, candidate) for item in row)
    if best_overlap >= MIN_VERTICAL_OVERLAP_RATIO:
        return True
    median_h = row_median_height(row)
    center_gap = abs(row_mean_center_y(row) - candidate.center_y)
    return center_gap <= MAX_CENTER_Y_RATIO * median_h


def group_lines_into_rows(lines: Sequence[TextLine]) -> list[list[TextLine]]:
    """Cluster word boxes by vertical overlap after undoing page skew."""
    if not lines:
        return []

    skew = estimate_page_skew_degrees(lines)
    origin_x = sum(sum(x for x, _y in item.box) for item in lines) / float(
        4 * len(lines)
    )
    origin_y = sum(sum(y for _x, y in item.box) for item in lines) / float(
        4 * len(lines)
    )
    origin = (origin_x, origin_y)
    flat = [flatten_box(item, skew_degrees=skew, origin=origin) for item in lines]

    ordered = sorted(flat, key=lambda item: (item.center_y, item.left))
    rows: list[list[FlatBox]] = []
    for item in ordered:
        best_index = -1
        best_score = -1.0
        for index, row in enumerate(rows):
            if not fits_text_row(row, item):
                continue
            score = max(vertical_overlap_ratio(existing, item) for existing in row)
            score += 1.0 / (1.0 + abs(row_mean_center_y(row) - item.center_y))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            rows.append([item])
        else:
            rows[best_index].append(item)

    rows = [sorted(row, key=lambda item: item.left) for row in rows]
    rows.sort(key=row_mean_center_y)
    return [[item.line for item in row] for row in rows]


def render_documents(
    documents: Sequence[RecognizedDocument],
    *,
    spell: bool = True,
    fix_caps: bool = True,
    strip_junk: bool = False,
) -> str:
    """One file: plain text. Several files: name headers between blocks."""
    if len(documents) == 1:
        return document_text(
            documents[0],
            spell=spell,
            fix_caps=fix_caps,
            strip_junk=strip_junk,
        )

    blocks = [
        f"=== {document.path.name} ===\n"
        f"{document_text(document, spell=spell, fix_caps=fix_caps, strip_junk=strip_junk)}"
        for document in documents
    ]
    return "\n\n".join(blocks)


def document_text(
    document: RecognizedDocument,
    *,
    spell: bool = True,
    fix_caps: bool = True,
    strip_junk: bool = False,
) -> str:
    """Join words on the same visual row with spaces; rows with newlines."""
    rows = group_lines_into_rows(document.lines)
    rendered = [
        " ".join(part.text for part in row if part.text)
        for row in rows
    ]
    rendered = [line for line in rendered if line]
    if not rendered:
        return t("msg.no_text")
    text = "\n".join(rendered)
    # Always normalize quotes; other polish follows flags.
    return correct_text(
        text,
        spell=spell,
        fix_caps=fix_caps,
        strip_junk=strip_junk,
    )


def render_debug(documents: Sequence[RecognizedDocument]) -> str:
    """Human-readable debug dump: index, confidence, box corners."""
    blocks: list[str] = []
    for document in documents:
        rows = [t("msg.debug_file", path=document.path)]
        if not document.lines:
            rows.append(t("msg.no_text"))
        for number, line in enumerate(document.lines, start=1):
            rows.append(f"{number}. [{line.confidence:.2f}] {line.text}")
            rows.append(t("msg.debug_coords", box=format_box(line.box)))
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks)


def format_box(box: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"({x}, {y})" for x, y in box)


def debug_output_path(output: Path) -> Path:
    """result.txt -> result.debug.txt next to the main output."""
    suffix = output.suffix or ".txt"
    return output.with_name(f"{output.stem}.debug{suffix}")


def save_text(path: Path, text: str) -> None:
    """Write UTF-8 with a trailing newline. Replaces an existing file."""
    directory = path.parent
    if str(directory) not in ("", ".") and not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write(text)
        if not text.endswith("\n"):
            output.write("\n")


def print_result(text: str, output: Path) -> None:
    print(t("msg.recognized_header"))
    print(text)
    print(t("msg.text_saved", path=output))


def configure_console() -> None:
    """Ask Windows consoles for UTF-8. Ignore failure and continue."""
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
        print(t("msg.error_prefix", error=error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
