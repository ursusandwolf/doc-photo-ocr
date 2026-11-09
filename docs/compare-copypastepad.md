# Comparison: OCR-CopyPastePad

Source: [FlyingFathead/OCR-CopyPastePad](https://github.com/FlyingFathead/OCR-CopyPastePad) (`OCR-CopyPastePad.py`, ~690 lines).

**License:** the upstream repo has **no LICENSE** file. Do not copy their source. Reimplement UX ideas in our own code; ask the author only if verbatim reuse is ever required.

## Feature map

| Feature | CopyPastePad | doc-photo-ocr | Touch points here |
|---|---|---|---|
| EasyOCR | Button / in-process Reader | Default via warm worker | [`ocr.py`](../ocr.py), [`gui.py`](../gui.py) |
| Tesseract | Default button + PATH binary | Not shipped (Strategy roadmap only) | Future `OcrEngine` protocol |
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

We keep modules + a long-lived worker. Dual-engine / Tesseract stays in the README “Possible evolution” section until a second backend is scheduled — not in PR1.

## QC on `resources/` fixtures

Method: same PNGs; ours via `ocr.py --lang ru+en --no-preprocess`; CPP-like EasyOCR via in-process `readtext`; Tesseract via `pytesseract` when installed. Synthetic Arial pages. Full snippets: [`qc-copypastepad-log.md`](qc-copypastepad-log.md).

| Fixture | Ours | CPP-like Tesseract | CPP-like EasyOCR |
|---|---|---|---|
| `zayavlenie_otpusk.png` | Body OK; «СеверТорг»/№/HR weak | Not installed locally | Similar body; still breaks guillemets/№ |
| `spravka_s_mesta_raboty.png` | Body OK; ИНН→ИН, e-mail broken | Not installed locally | Slightly better ИНН/СПРАВКА; e-mail still broken |
| `nakladnaya_tovarnaya.png` | Weak (КОВАРНАЯ, table qty) | Not installed locally | Same class of table/№ errors |

Conclusion: raw EasyOCR quality is close either way; our polish helps some tokens but does not fix tables/№. Tesseract comparison needs `pytesseract` + `tesseract.exe` on the machine (deferred with dual-engine).

## Adopted vs rejected

**Adopted (PR1):** clipboard paste, copy result, optional invert.

**Rejected for now:** mandatory Tesseract in the Windows zip; CopyPastePad hard binarize as default; reloading EasyOCR on every language change on the UI thread; monolithic GUI=OCR style.
