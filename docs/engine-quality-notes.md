# Engine quality notes (manual QC)

**Date:** 2026-11-10  
**Status:** Tesseract — good enough on current samples; EasyOCR — needs deeper analysis.

## What was checked

Manual recognition on the same images with both engines (shared preprocess + postprocess).  
Layout/`plain` fixes are in place; remaining gap is **token quality**, not reading order.

## Result

| Engine | Manual verdict |
|---|---|
| **Tesseract** (`--engine tesseract`) | Acceptable / good text on the checked photos (digits, `№`, quotes, continuous Russian lines). |
| **EasyOCR** (default) | Noticeably more junk / substitutions on the **same** images. |

Conclusion for product use today: prefer **Tesseract** for these document photos until EasyOCR is improved or ensemble exists.

## Why EasyOCR likely underperforms here (code-level hypotheses)

Not yet proven with A/B metrics — backlog for analysis. Ordered by likelihood:

1. **Detect → cluster → recognize-per-row** ([`ocr.py`](../ocr.py) `read_lines`)  
   CRAFT boxes are clustered by our geometry, then each row is re-recognized as `free_list` crops. Bad / fragmented detect boxes → wrong crops → junk tokens even when layout order is correct.  
   Tesseract uses `image_to_data` word boxes + native `line_num` seed ([`engines.py`](../engines.py) `group_tesseract_native_rows`).

2. **No Tesseract-style page segmentation**  
   EasyOCR has no PSM / block structure; everything depends on CRAFT + our clustering. Dense print and small glyphs suffer.

3. **CRNN `cyrillic_g2` vs tessdata `rus+eng`**  
   Different confusion profiles (`Ng`/`©`/`$`, Latin/Cyrillic mixes). Postprocess mitigates some of this for both engines, but cannot recover badly recognized crops.

4. **Preprocess tuned as “photo gray”** ([`preprocess.py`](../preprocess.py))  
   CLAHE + light denoise helps phone photos; for flatter scans Tesseract often needs less “help”, while EasyOCR may still over-segment after contrast boost.

5. **Default language / dictionary path**  
   EasyOCR `ru` vs `ru+en` changes recognition dictionaries; wrong mix increases Latin lookalikes inside Cyrillic words.

## Analysis backlog (EasyOCR)

- [ ] Same image, `--engine easyocr` vs `tesseract`, save `--debug` and compare low-conf (`[!]` ) tokens.
- [ ] Dump CRAFT `horizontal_list` / `free_list` counts vs Tesseract word count.
- [ ] Try EasyOCR with `--no-preprocess` and with `--lang ru+en` on one failing sample.
- [ ] Spot-check whether junk lines correlate with multi-box rows (cluster merge) or single-box misreads.
- [ ] Only after that: ensemble / confidence re-OCR / fine-tune (see quality triage plan).

## Related

- Layout work: `plain` / `table`, residual skew, postprocess lookalikes — separate from this engine gap.
- Ensemble still not implemented (README).
