# Comparison: OCR-CopyPastePad

Source: [FlyingFathead/OCR-CopyPastePad](https://github.com/FlyingFathead/OCR-CopyPastePad) (`OCR-CopyPastePad.py`, ~690 lines).

**License:** the upstream repo has **no LICENSE** file. Do not copy their source. Reimplement UX ideas in our own code; ask the author only if verbatim reuse is ever required.

## Feature map

| Feature | CopyPastePad | doc-photo-ocr | Touch points here |
|---|---|---|---|
| EasyOCR | Button / in-process Reader | Default via warm worker | [`ocr.py`](../ocr.py), [`gui.py`](../gui.py) |
| Tesseract | Default button + PATH binary | Optional `--engine tesseract` / GUI combo | [`engines.py`](../engines.py) |
| Soft/hard preprocess | Hard: ×3 upscale, adaptive thresh, dilate | Soft deskew/CLAHE; no hard binarize by default | [`preprocess.py`](../preprocess.py) |
| Clipboard paste image | Ctrl+V / Shift+Insert (`ImageGrab`) | Adopted (PR1) | [`gui.py`](../gui.py) |
| Copy text out | Text pane (manual select) | Copy-result button (PR1) | [`gui.py`](../gui.py) |
| Invert colors | Button + auto `should_invert` | Optional checkbox / `--invert` (PR1); no auto | [`preprocess.py`](../preprocess.py), [`ocr.py`](../ocr.py), [`gui.py`](../gui.py) |
| Crop / ROI | Canvas rectangle | Deferred (wave 2) | — |
| Side-by-side preview | PanedWindow image+text | Deferred (wave 2) | — |
| Batch folder / DnD | No | Yes | [`gui.py`](../gui.py), [`image_paths.py`](../image_paths.py) |
| Offline models | EasyOCR may download on start | Explicit `--download-models` only | [`ocr.py`](../ocr.py) |
| Spell / caps / junk | No | [`spellcheck.py`](../spellcheck.py) | — |
| Themes / i18n | No | Themes + ru/en UI | [`gui.py`](../gui.py), [`i18n.py`](../i18n.py) |

## Architecture

CopyPastePad is a single GUI class with OCR calls on the UI thread (`update_idletasks`). Engines are two buttons, not a Strategy interface.

We keep modules + a long-lived worker. Backends implement `OcrBackend` in [`engines.py`](../engines.py); EasyOCR remains default. Auto score / “retry other engine” is still phase 2.

## QC on `resources/` fixtures

Method: same PNGs; ours via `ocr.py --lang ru+en --no-preprocess`; CPP-like EasyOCR via in-process `readtext`; Tesseract via `pytesseract` when installed. Synthetic Arial pages. Full snippets: [`qc-copypastepad-log.md`](qc-copypastepad-log.md).

| Fixture | Ours (`--engine easyocr`) | `--engine tesseract` | Raw EasyOCR `readtext` |
|---|---|---|---|
| `zayavlenie_otpusk.png` | Body OK; «СеверТорг»/№/HR weak | Binary missing on this PC (`err.no_tesseract`) | Similar body; guillemets/№ still break |
| `spravka_s_mesta_raboty.png` | Body OK; ИНН→ИН, e-mail broken | Same: needs `tesseract.exe` on PATH | Slightly better ИНН/СПРАВКА |
| `nakladnaya_tovarnaya.png` | Weak (КОВАРНАЯ, table qty) | Same: needs `tesseract.exe` on PATH | Same class of table/№ errors |

Conclusion: EasyOCR quality is close with/without our polish on tokens; tables/№ stay hard. Tesseract path is wired (`engines.TesseractEngine`); install UB-Mannheim + rus/eng to fill the middle column. Re-run `docs/_qc_compare_run.py` after installing the binary.

## Adopted vs rejected

**Adopted (PR1):** clipboard paste, copy result, optional invert.

**Rejected for now:** mandatory Tesseract in the Windows zip; CopyPastePad hard binarize as default; reloading EasyOCR on every language change on the UI thread; monolithic GUI=OCR style.
