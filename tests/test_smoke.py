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
import spellcheck

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

    def test_normalize_engine(self) -> None:
        from engines import normalize_engine

        self.assertEqual(normalize_engine("EasyOCR"), "easyocr")
        self.assertEqual(normalize_engine("tesseract"), "tesseract")
        with self.assertRaises(ocr.OcrError):
            normalize_engine("paddle")


class EngineSmokeTests(unittest.TestCase):
    def test_tesseract_lang_string(self) -> None:
        from engines import tesseract_lang_string

        self.assertEqual(tesseract_lang_string(["ru"]), "rus")
        self.assertEqual(tesseract_lang_string(["ru", "en"]), "rus+eng")

    def test_parse_tesseract_data_builds_boxes(self) -> None:
        from engines import parse_tesseract_data

        data = {
            "text": ["", "Hello", "мир"],
            "conf": ["-1", "95", "80"],
            "left": [0, 10, 50],
            "top": [0, 20, 20],
            "width": [0, 40, 30],
            "height": [0, 12, 12],
        }
        lines = parse_tesseract_data(data)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, "Hello")
        self.assertAlmostEqual(lines[0].confidence, 0.95)
        self.assertEqual(lines[0].box[0], (10, 20))
        self.assertEqual(lines[1].text, "мир")

    def test_tesseract_missing_binary_raises(self) -> None:
        from engines import TesseractEngine, ensure_tesseract_available

        try:
            ensure_tesseract_available()
        except ocr.OcrError as error:
            self.assertIn("Tesseract", str(error))
            return
        # Binary present: warm must succeed.
        engine = TesseractEngine(["ru", "en"])
        engine.warm()

    def test_resolve_tesseract_cmd_reads_pref_file(self) -> None:
        import engines
        from engines import resolve_tesseract_cmd

        fake_exe = Path(self._temp_dir()) / "tesseract.exe"
        fake_exe.write_bytes(b"")
        original = engines.TESSERACT_CMD_FILE
        engines.TESSERACT_CMD_FILE = Path(self._temp_dir()) / "tesseract_cmd.txt"
        try:
            engines.TESSERACT_CMD_FILE.write_text(str(fake_exe) + "\n", encoding="utf-8")
            self.assertEqual(resolve_tesseract_cmd(), fake_exe)
        finally:
            engines.TESSERACT_CMD_FILE = original

    def test_is_tesseract_ready_false_without_langs(self) -> None:
        import engines
        from tesseract_setup import is_tesseract_ready

        fake_root = Path(self._temp_dir())
        fake_exe = fake_root / "tesseract.exe"
        fake_exe.write_bytes(b"")
        original = engines.TESSERACT_CMD_FILE
        engines.TESSERACT_CMD_FILE = fake_root / "tesseract_cmd.txt"
        try:
            engines.TESSERACT_CMD_FILE.write_text(str(fake_exe) + "\n", encoding="utf-8")
            self.assertFalse(is_tesseract_ready())
        finally:
            engines.TESSERACT_CMD_FILE = original

    def _temp_dir(self) -> str:
        import tempfile

        if not hasattr(self, "_tmpdir"):
            self._tmpdir = tempfile.mkdtemp(prefix="dococr-tess-test-")
        return self._tmpdir


class OfflineGuardTests(unittest.TestCase):
    def test_offline_session_blocks_urlopen(self) -> None:
        import urllib.request

        from offline_guard import offline_session

        with offline_session():
            with self.assertRaises(ocr.OcrError) as ctx:
                urllib.request.urlopen("https://example.com/")
        self.assertIn("offline", str(ctx.exception).lower())

    def test_create_reader_without_download_does_not_call_urlopen(self) -> None:
        import urllib.request

        calls: list[object] = []
        original = urllib.request.urlopen

        def tracking(*args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        urllib.request.urlopen = tracking  # type: ignore[assignment]
        try:
            if ocr.local_models_present():
                ocr.create_reader(["ru"], allow_download=False)
            else:
                with self.assertRaises(ocr.OcrError):
                    ocr.create_reader(["ru"], allow_download=False)
            self.assertEqual(calls, [])
        finally:
            urllib.request.urlopen = original  # type: ignore[assignment]

    def test_tesseract_pins_are_github_official(self) -> None:
        from tesseract_pins import (
            TESSDATA_ENG_URL,
            TESSDATA_RUS_URL,
            TESSERACT_RELEASE_PAGE,
            TESSERACT_SETUP_SHA256,
            TESSERACT_SETUP_URL,
        )

        for url in (
            TESSERACT_SETUP_URL,
            TESSERACT_RELEASE_PAGE,
            TESSDATA_ENG_URL,
            TESSDATA_RUS_URL,
        ):
            self.assertTrue(url.startswith("https://github.com/tesseract-ocr/"))
        self.assertEqual(len(TESSERACT_SETUP_SHA256), 64)

    def test_tesseract_checksum_rejects_mismatch(self) -> None:
        import tempfile
        from pathlib import Path

        from errors import OcrError
        from tesseract_setup import _verify_sha256

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.exe"
            path.write_bytes(b"not-the-real-installer")
            with self.assertRaises(OcrError) as ctx:
                _verify_sha256(path, "0" * 64)
            self.assertIn("SHA-256", str(ctx.exception))
            self.assertFalse(path.exists())

    def test_tesseract_setup_import_is_offline(self) -> None:
        import urllib.request

        calls: list[object] = []
        original = urllib.request.urlopen

        def tracking(*args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            raise AssertionError("unexpected network")

        urllib.request.urlopen = tracking  # type: ignore[assignment]
        try:
            import importlib

            import tesseract_setup

            importlib.reload(tesseract_setup)
            self.assertFalse(tesseract_setup.is_tesseract_ready())
            self.assertEqual(calls, [])
        finally:
            urllib.request.urlopen = original  # type: ignore[assignment]


class PreprocessSmokeTests(unittest.TestCase):
    def test_full_frame_page_keeps_size(self) -> None:
        page = np.full((900, 700, 3), 255, np.uint8)
        cv2.rectangle(page, (2, 2), (697, 897), (0, 0, 0), 3)
        prepared = ocr.preprocess(page)
        self.assertEqual(prepared.shape, (900, 700))
        self.assertIsNone(ocr.find_document_corners(page))

    def test_local_models_check_matches_disk(self) -> None:
        self.assertEqual(ocr.local_models_present(), any(MODELS_DIR.glob("*.pth")))

    def test_invert_image_flips_pixels(self) -> None:
        from preprocess import invert_image

        page = np.zeros((4, 4, 3), np.uint8)
        page[0, 0] = (10, 20, 30)
        flipped = invert_image(page)
        self.assertEqual(tuple(flipped[0, 0]), (245, 235, 225))
        self.assertTrue(np.array_equal(invert_image(flipped), page))


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
        text = ocr.document_text(document, spell=False)
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
        text = ocr.document_text(ocr.RecognizedDocument(Path("x.png"), lines), spell=False)
        self.assertEqual(
            text,
            "Прошу предоставить мне\n15.07.2025 по 28.07.2025",
        )

    def test_skewed_boxes_keep_reading_order(self) -> None:
        # Real EasyOCR parallelograms after preprocess on zayavlenie_otpusk.png.
        lines = (
            ocr.TextLine("отпуск", 1.0, ((715, 436), (812, 423), (816, 457), (719, 471))),
            ocr.TextLine("оплачиваемый", 1.0, ((516, 475), (716, 438), (720, 466), (521, 504))),
            ocr.TextLine("дней)", 1.0, ((812, 465), (895, 452), (900, 488), (816, 501))),
            ocr.TextLine("жкалендарных", 0.68, ((633, 503), (817, 470), (822, 502), (638, 534))),
            ocr.TextLine("ежегодный", 0.69, ((366, 504), (516, 473), (521, 503), (371, 534))),
            ocr.TextLine("мне", 1.0, ((312, 513), (367, 504), (370, 531), (315, 539))),
            ocr.TextLine("включительно", 1.0, ((397, 544), (588, 513), (592, 540), (401, 571))),
            ocr.TextLine("предоставить", 1.0, ((122, 548), (309, 513), (315, 544), (127, 579))),
            ocr.TextLine("2025", 1.0, ((324, 553), (399, 542), (404, 577), (329, 588))),
            ocr.TextLine("Прошу", 1.0, ((31, 558), (128, 550), (129, 588), (33, 596))),
            ocr.TextLine("(14", 1.0, ((591, 503), (637, 503), (637, 539), (591, 539))),
            ocr.TextLine("по 28.07", 0.77, ((212, 558), (328, 558), (328, 606), (212, 606))),
            ocr.TextLine("15.07", 0.70, ((65, 593), (141, 593), (141, 633), (65, 633))),
        )
        text = ocr.document_text(ocr.RecognizedDocument(Path("x.png"), lines), spell=False)
        self.assertEqual(
            text.splitlines()[0],
            "Прошу предоставить мне ежегодный оплачиваемый отпуск",
        )
        self.assertRegex(
            text.splitlines()[1],
            r"15\.07.*по 28\.07.*включительно.*\(14.*дней\)",
        )

    def test_detection_placeholders_cluster_into_rows(self) -> None:
        horizontal = (
            [10, 80, 100, 128],
            [90, 200, 100, 128],
            [10, 120, 136, 164],
            [130, 290, 136, 164],
        )
        placeholders = ocr.detection_placeholders(horizontal, [])
        rows = ocr.group_lines_into_rows(placeholders)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), 2)
        self.assertEqual(len(rows[1]), 2)
        self.assertEqual(ocr.box_as_free_list_item(rows[0][0].box)[0], [10, 100])

    def test_parse_easyocr_results_skips_empty(self) -> None:
        raw = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "Hello", 0.9),
            ([[20, 0], [30, 0], [30, 10], [20, 10]], "  ", 0.5),
        ]
        lines = ocr.parse_easyocr_results(raw)
        self.assertEqual([item.text for item in lines], ["Hello"])


class SpellcheckSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        spellcheck.reset_spellcheckers_cache()

    def test_fixes_ocr_typo_and_keeps_dates_codes(self) -> None:
        text = "14 жкалендарных дней\n15.07.2025 HR-2025-078"
        fixed = spellcheck.correct_text(text)
        self.assertIn("календарных", fixed)
        self.assertNotIn("жкалендарных", fixed)
        self.assertIn("15.07.2025", fixed)
        self.assertIn("HR-2025-078", fixed)

    def test_fixes_unexpected_caps_in_lowercase_words(self) -> None:
        text = "согласно графику отпусков на 2025 гОД-"
        fixed = spellcheck.correct_text(
            text,
            spell=False,
            fix_caps=True,
            strip_junk=False,
        )
        self.assertIn("год-", fixed)
        self.assertNotIn("гОД", fixed)
        self.assertEqual(
            spellcheck.correct_text(
                text,
                spell=False,
                fix_caps=False,
                strip_junk=False,
            ),
            text,
        )
        self.assertFalse(spellcheck.has_unexpected_caps("ЗАЯВЛЕНИЕ"))
        self.assertFalse(spellcheck.has_unexpected_caps("Петрова"))
        self.assertTrue(spellcheck.has_unexpected_caps("гОД"))

    def test_strips_junk_characters(self) -> None:
        text = "Директору <Северторг ~? отдел\nгод-"
        fixed = spellcheck.correct_text(
            text,
            spell=False,
            fix_caps=False,
            strip_junk=True,
        )
        self.assertEqual(fixed, "Директору Северторг отдел\nгод")
        self.assertNotIn("<", fixed)
        self.assertNotIn("~", fixed)
        cluster = spellcheck.strip_junk_text('текст !"№;%::?*( ~`:;"|/\\? хвост')
        self.assertEqual(cluster, "текст хвост")
        self.assertEqual(spellcheck.strip_junk_text("таб. № 1042"), "таб. № 1042")
        self.assertEqual(spellcheck.strip_junk_text("100%"), "100%")
        self.assertEqual(spellcheck.strip_junk_text("что?"), "что?")
        self.assertEqual(spellcheck.strip_junk_text("а: б; в!"), "а: б; в!")
        self.assertEqual(spellcheck.strip_junk_text("a  ~  b"), "a b")

    def test_keeps_email_and_hyphenated_email_label(self) -> None:
        line = "тел. +7 (495) 123-45-67 e-mail: anna@example.com"
        fixed = spellcheck.correct_text(line, spell=True, fix_caps=True, strip_junk=True)
        self.assertIn("e-mail:", fixed)
        self.assertIn("anna@example.com", fixed)
        self.assertIn("+7 (495) 123-45-67", fixed)
        self.assertNotIn("email:", fixed)

    def test_normalizes_guillemets(self) -> None:
        self.assertEqual(
            spellcheck.normalize_guillemets('фирма "СеверТорг"'),
            "фирма «СеверТорг»",
        )
        self.assertEqual(
            spellcheck.normalize_guillemets("фирма „СеверТорг“"),
            "фирма «СеверТорг»",
        )
        kept = spellcheck.correct_text(
            "фирма «СеверТорг»",
            spell=False,
            fix_caps=False,
            strip_junk=True,
        )
        self.assertEqual(kept, "фирма «СеверТорг»")

    def test_document_text_respects_no_spell(self) -> None:
        lines = (
            ocr.TextLine(
                "жкалендарных",
                0.9,
                ((10, 100), (120, 100), (120, 130), (10, 130)),
            ),
        )
        document = ocr.RecognizedDocument(Path("x.png"), lines)
        self.assertEqual(ocr.document_text(document, spell=False), "жкалендарных")
        self.assertEqual(ocr.document_text(document, spell=True), "календарных")


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
            code = ocr.main(
                [
                    str(image_path),
                    "-o",
                    str(output),
                    "--no-preprocess",
                    "--lang",
                    "ru+en",
                ]
            )
            self.assertEqual(code, 0)
            text = output.read_text(encoding="utf-8")

        lowered = text.lower()
        self.assertIn("договор", lowered)
        self.assertTrue(
            "contract" in lowered or "a4" in lowered or "а4" in lowered
        )

    def test_on_row_streams_before_final_text(self) -> None:
        image_path = ROOT / "resources" / "zayavlenie_otpusk.png"
        if not image_path.exists():
            self.skipTest("fixture missing")
        from engines import create_engine

        engine = create_engine("easyocr", ["ru", "en"], allow_download=False)
        engine.warm()
        streamed: list[str] = []
        row_counts: list[int] = []
        document = ocr.recognize_file(
            engine,
            image_path,
            preprocess_image=True,
            on_row=streamed.append,
            on_rows=row_counts.append,
        )
        final = ocr.document_text(document)
        self.assertEqual(row_counts, [len(streamed)])
        non_empty = [row for row in streamed if row]
        self.assertGreaterEqual(len(non_empty), 3)
        self.assertTrue(any("Прошу" in row for row in non_empty))
        self.assertIn("Прошу предоставить", final)

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
