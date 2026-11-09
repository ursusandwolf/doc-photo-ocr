"""Local OCR for A4 document photos.

Run path:

1. Parse CLI arguments.
2. Either download models once (--download-models) or recognize photos.
3. Check that each path exists and the format is supported.
4. Load the image (including Cyrillic paths on Windows).
5. Preprocess unless --no-preprocess is set.
6. Run EasyOCR on CPU with local weights.
7. Print text and write result.txt (or one .txt per file with --per-file).

Inputs may be files and/or folders. A folder takes first-level images;
--recursive also walks nested directories.

Recognition always builds Reader with download_enabled=False.
Network download is allowed only in the --download-models branch.
Image prep lives in preprocess.py — no network and no OCR there.
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

from i18n import get_ui_lang, set_ui_lang, t
from image_paths import SUPPORTED_EXTENSIONS
from image_paths import PathError as ImagePathError
from image_paths import collect_images as collect_image_paths
from preprocess import find_document_corners, preprocess

DEFAULT_LANGUAGES = "ru+en"
DEFAULT_OUTPUT = Path("result.txt")
MODEL_DIR = Path(__file__).resolve().parent / "models"
# Line clustering: prefer vertical overlap; center distance is a fallback.
MIN_VERTICAL_OVERLAP_RATIO = 0.5
MAX_CENTER_Y_RATIO = 0.35

# EasyOCR stores detection/recognition weights under models/.
# Without .pth files we fail early with a clear offline error.
MODEL_FILE_GLOB = "*.pth"


class OcrError(Exception):
    """User-visible failure without a traceback dump."""


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
        "--download-models",
        action="store_true",
        help=t("cli.download_help"),
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
    """Run either model download or recognition."""
    set_ui_lang(getattr(args, "ui", get_ui_lang()))
    languages = parse_languages(args.lang)

    if args.download_models:
        if args.images:
            raise OcrError(t("err.download_with_images"))
        download_models(languages)
        return

    if not args.images:
        raise OcrError(t("err.need_input"))

    image_paths = collect_images(args.images, recursive=args.recursive)
    if not image_paths:
        raise OcrError(t("err.no_images"))

    for image_path in image_paths:
        ensure_supported_image(image_path)

    print(t("msg.files_count", count=len(image_paths)))
    reader = create_reader(languages, allow_download=False)
    documents: list[RecognizedDocument] = []
    for image_path in image_paths:
        print(t("msg.processing", path=image_path))
        documents.append(
            recognize_file(
                reader,
                image_path,
                preprocess_image=not args.no_preprocess,
            )
        )

    if args.per_file:
        write_per_file_results(documents, debug=args.debug)
        return

    text = render_documents(documents)
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
) -> None:
    """Write document.png -> document.txt next to each input."""
    if not documents:
        raise OcrError(t("err.no_docs"))

    for document in documents:
        output = document.path.with_suffix(".txt")
        text = document_text(document)
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


def recognize_file(reader: Any, path: Path, *, preprocess_image: bool) -> RecognizedDocument:
    """Read one file and return lines in reading order."""
    image = load_image(path)
    try:
        prepared = preprocess(image) if preprocess_image else image
    except cv2.error as error:
        raise OcrError(t("err.preprocess", name=path.name, error=error)) from error
    lines = read_lines(reader, prepared, path)
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


def read_lines(reader: Any, image: np.ndarray, path: Path) -> list[TextLine]:
    """Run OCR. Word boxes are kept separate; document_text joins them into lines."""
    try:
        with silence_library_noise():
            # paragraph=False keeps boxes; we rebuild lines ourselves for stable order.
            raw_rows = reader.readtext(image, detail=1, paragraph=False)
    except Exception as error:
        raise OcrError(t("err.recognize", name=path.name, error=error)) from error

    lines: list[TextLine] = []
    for box, text, confidence in raw_rows:
        points = tuple((int(x), int(y)) for x, y in box)
        cleaned = str(text).strip()
        if cleaned:
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


def vertical_overlap_ratio(a: TextLine, b: TextLine) -> float:
    """overlap_height / min(heights). 1.0 means fully nested on the Y axis."""
    top = max(box_top(a), box_top(b))
    bottom = min(box_bottom(a), box_bottom(b))
    overlap = max(0, bottom - top)
    return overlap / float(min(box_height(a), box_height(b)))


def row_median_height(row: Sequence[TextLine]) -> float:
    heights = sorted(box_height(item) for item in row)
    mid = len(heights) // 2
    if not heights:
        return 1.0
    if len(heights) % 2:
        return float(heights[mid])
    return (heights[mid - 1] + heights[mid]) / 2.0


def row_mean_center_y(row: Sequence[TextLine]) -> float:
    return sum(box_center_y(item) for item in row) / float(len(row))


def fits_text_row(row: Sequence[TextLine], candidate: TextLine) -> bool:
    """True when candidate belongs on this visual text line."""
    best_overlap = max(vertical_overlap_ratio(item, candidate) for item in row)
    if best_overlap >= MIN_VERTICAL_OVERLAP_RATIO:
        return True
    median_h = row_median_height(row)
    center_gap = abs(row_mean_center_y(row) - box_center_y(candidate))
    return center_gap <= MAX_CENTER_Y_RATIO * median_h


def group_lines_into_rows(lines: Sequence[TextLine]) -> list[list[TextLine]]:
    """Cluster word boxes by vertical overlap, then left-to-right in each row."""
    if not lines:
        return []

    ordered = sorted(lines, key=lambda item: (box_center_y(item), box_left(item)))
    rows: list[list[TextLine]] = []
    for item in ordered:
        best_index = -1
        best_score = -1.0
        for index, row in enumerate(rows):
            if not fits_text_row(row, item):
                continue
            score = max(vertical_overlap_ratio(existing, item) for existing in row)
            # Prefer stronger overlap; break ties with closer row center.
            score += 1.0 / (1.0 + abs(row_mean_center_y(row) - box_center_y(item)))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            rows.append([item])
        else:
            rows[best_index].append(item)

    rows = [sorted(row, key=box_left) for row in rows]
    rows.sort(key=row_mean_center_y)
    return rows


def render_documents(documents: Sequence[RecognizedDocument]) -> str:
    """One file: plain text. Several files: name headers between blocks."""
    if len(documents) == 1:
        return document_text(documents[0])

    blocks = [
        f"=== {document.path.name} ===\n{document_text(document)}"
        for document in documents
    ]
    return "\n\n".join(blocks)


def document_text(document: RecognizedDocument) -> str:
    """Join words on the same visual row with spaces; rows with newlines."""
    rows = group_lines_into_rows(document.lines)
    rendered = [
        " ".join(part.text for part in row if part.text)
        for row in rows
    ]
    rendered = [line for line in rendered if line]
    if not rendered:
        return t("msg.no_text")
    return "\n".join(rendered)


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
