"""Минимальные проверки MVP: CLI, предобработка, офлайн-распознавание.

Полный OCR запускается только если папка models уже заполнена.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ocr

MODELS_DIR = ROOT / "models"
SAMPLE_IMAGE = ROOT / "samples" / "sample_contract.png"


class CliSmokeTests(unittest.TestCase):
    def test_missing_file_returns_error(self) -> None:
        code = ocr.main([str(ROOT / "нет-такого-файла.jpg")])
        self.assertEqual(code, 1)

    def test_unsupported_extension_returns_error(self) -> None:
        code = ocr.main([str(ROOT / "ocr.py")])
        self.assertEqual(code, 1)

    def test_download_models_rejects_image_paths(self) -> None:
        code = ocr.main(["--download-models", str(SAMPLE_IMAGE)])
        self.assertEqual(code, 1)

    def test_parse_languages(self) -> None:
        self.assertEqual(ocr.parse_languages("ru+en,ru"), ["ru", "en"])


class PreprocessSmokeTests(unittest.TestCase):
    def test_full_frame_page_keeps_size(self) -> None:
        page = np.full((900, 700, 3), 255, np.uint8)
        cv2.rectangle(page, (2, 2), (697, 897), (0, 0, 0), 3)
        prepared = ocr.preprocess(page)
        self.assertEqual(prepared.shape, (900, 700))
        self.assertIsNone(ocr.find_document_corners(page))

    def test_local_models_check_matches_disk(self) -> None:
        self.assertEqual(ocr.local_models_present(), any(MODELS_DIR.glob("*.pth")))


class DropPathSmokeTests(unittest.TestCase):
    def test_decode_drop_path_str_and_bytes(self) -> None:
        import gui

        sample = ROOT / "resources" / "zayavlenie_otpusk.png"
        self.assertEqual(gui.decode_drop_path(str(sample)), sample)
        self.assertEqual(gui.decode_drop_path(str(sample).encode("utf-8")), sample)

    def test_collect_images_via_image_paths(self) -> None:
        from image_paths import collect_images

        folder = ROOT / "resources"
        names = [item.name for item in collect_images([folder])]
        self.assertIn("zayavlenie_otpusk.png", names)


class UiLangSmokeTests(unittest.TestCase):
    def test_ui_english_help_and_errors(self) -> None:
        from i18n import set_ui_lang, t

        set_ui_lang("en")
        self.assertIn("Recognize text", t("cli.description"))
        code = ocr.main(["--ui", "en"])
        self.assertEqual(code, 1)
        set_ui_lang("ru")
        self.assertIn("Распознаёт", t("cli.description"))


class TextLayoutSmokeTests(unittest.TestCase):
    def test_words_on_same_row_join_with_spaces(self) -> None:
        lines = (
            ocr.TextLine("Прошу", 0.9, ((10, 100), (80, 100), (80, 130), (10, 130))),
            ocr.TextLine("предоставить", 0.9, ((90, 102), (220, 102), (220, 132), (90, 132))),
            ocr.TextLine("отпуск", 0.9, ((230, 98), (320, 98), (320, 128), (230, 128))),
            ocr.TextLine("Согласно", 0.9, ((10, 160), (100, 160), (100, 190), (10, 190))),
            ocr.TextLine("графику", 0.9, ((110, 162), (200, 162), (200, 192), (110, 192))),
        )
        document = ocr.RecognizedDocument(Path("x.png"), lines)
        text = ocr.document_text(document)
        self.assertEqual(
            text,
            "Прошу предоставить отпуск\nСогласно графику",
        )

    def test_nearby_rows_do_not_mix_when_sorted_by_x(self) -> None:
        # Two close rows: loose center matching used to merge them and scramble order.
        lines = (
            ocr.TextLine("Прошу", 0.9, ((10, 100), (70, 100), (70, 128), (10, 128))),
            ocr.TextLine("предоставить", 0.9, ((80, 100), (200, 100), (200, 128), (80, 128))),
            ocr.TextLine("мне", 0.9, ((210, 100), (250, 100), (250, 128), (210, 128))),
            ocr.TextLine("15.07.2025", 0.9, ((10, 136), (120, 136), (120, 164), (10, 164))),
            ocr.TextLine("по", 0.9, ((130, 136), (160, 136), (160, 164), (130, 164))),
            ocr.TextLine("28.07.2025", 0.9, ((170, 136), (290, 136), (290, 164), (170, 164))),
        )
        text = ocr.document_text(ocr.RecognizedDocument(Path("x.png"), lines))
        self.assertEqual(
            text,
            "Прошу предоставить мне\n15.07.2025 по 28.07.2025",
        )


class BatchPathSmokeTests(unittest.TestCase):
    def test_collect_images_from_file_and_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            photo = folder / "page.png"
            photo.write_bytes(b"not-needed-for-collect")
            (folder / "skip.txt").write_text("x", encoding="utf-8")
            nested = folder / "sub"
            nested.mkdir()
            deep = nested / "deep.jpg"
            deep.write_bytes(b"x")

            flat = ocr.collect_images([photo, folder], recursive=False)
            self.assertEqual([item.name for item in flat], ["page.png"])

            deep_list = ocr.collect_images([folder], recursive=True)
            names = sorted(item.name for item in deep_list)
            self.assertEqual(names, ["deep.jpg", "page.png"])

    def test_per_file_output_name(self) -> None:
        path = Path("resources/zayavlenie_otpusk.png")
        self.assertEqual(path.with_suffix(".txt").name, "zayavlenie_otpusk.txt")


@unittest.skipUnless(
    MODELS_DIR.exists() and any(MODELS_DIR.glob("*.pth")),
    "модели ещё не скачаны: python ocr.py --download-models",
)
class OfflineOcrSmokeTests(unittest.TestCase):
    def test_recognizes_russian_and_english_words(self) -> None:
        image_path = SAMPLE_IMAGE if SAMPLE_IMAGE.exists() else self._make_temp_image()
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.txt"
            code = ocr.main([str(image_path), "-o", str(output), "--no-preprocess"])
            self.assertEqual(code, 0)
            text = output.read_text(encoding="utf-8")

        lowered = text.lower()
        self.assertIn("договор", lowered)
        self.assertTrue("contract" in lowered or "a4" in lowered)

    def _make_temp_image(self) -> Path:
        path = Path(tempfile.gettempdir()) / "ocr_smoke_contract.png"
        image = Image.new("RGB", (1240, 800), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 42)
        draw.text((100, 120), "Договор № 123 от 15.03.2025", fill="black", font=font)
        draw.text((100, 220), "Contract ID: A4-RU-EN-001", fill="black", font=font)
        image.save(path)
        return path


if __name__ == "__main__":
    unittest.main()
