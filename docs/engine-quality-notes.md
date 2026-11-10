# Engine quality notes (manual QC + A/B)

**Date:** 2026-11-10  
**Status:** Tesseract preferred for product on real office photos; EasyOCR A/B on synthetic `resources/*.png` drives pipeline tweaks below.

## What was checked

1. Manual recognition on the same photos with both engines (shared preprocess + postprocess).  
2. Offline A/B via [`docs/_qc_engine_ab.py`](_qc_engine_ab.py) on fixtures:
   - `zayavlenie_otpusk.png`
   - `spravka_s_mesta_raboty.png`
   - `nakladnaya_tovarnaya.png`  
   Cases: Tesseract `ru`+prep; EasyOCR `ru`+prep; EasyOCR `ru`+noprep; EasyOCR `ru+en`+prep. Spell/fix-caps off. Full table: [`qc-engine-ab-summary.md`](qc-engine-ab-summary.md).

Layout/`plain` fixes are in place; remaining gap is **token quality**, not reading order.

## Manual result (real photos)

| Engine | Manual verdict |
|---|---|
| **Tesseract** (`--engine tesseract`) | Acceptable / good text on the checked photos (digits, `№`, quotes, continuous Russian lines). |
| **EasyOCR** (default) | Noticeably more junk / substitutions on the **same** images. |

Conclusion for product use today: prefer **Tesseract** for these document photos until EasyOCR is improved or ensemble exists. Default engine unchanged.

## A/B metrics (synthetic fixtures, 2026-11-10)

| Image | Engine | Lang | Prep | Lines | Low conf | Low % | Mean conf |
|---|---|---|---|---:|---:|---:|---:|
| zayavlenie | tesseract | ru | yes | 57 | 31 | 54.4 | 0.526 |
| zayavlenie | easyocr | ru | yes | 53 | 6 | 11.3 | 0.895 |
| zayavlenie | easyocr | ru | **no** | **17** | 3 | 17.6 | 0.821 |
| zayavlenie | easyocr | ru+en | yes | 53 | 7 | 13.2 | 0.897 |
| spravka | tesseract | ru | yes | 87 | 4 | 4.6 | 0.874 |
| spravka | easyocr | ru | yes | 32 | 10 | 31.2 | 0.809 |
| spravka | easyocr | ru | **no** | 27 | **4** | **14.8** | 0.820 |
| spravka | easyocr | ru+en | yes | 32 | 9 | 28.1 | 0.831 |
| nakladnaya | tesseract | ru | yes | 80 | 12 | 15.0 | 0.821 |
| nakladnaya | easyocr | ru | yes | 31 | 8 | 25.8 | 0.804 |
| nakladnaya | easyocr | ru | no | 30 | 8 | 26.7 | 0.788 |
| nakladnaya | easyocr | ru+en | yes | 31 | **5** | **16.1** | **0.848** |

### Facts (not hypotheses)

1. **Preprocess / detect:** EasyOCR with full prep over-fragments (`zayavlenie` 53 lines vs 17 without prep). Noprep text is cleaner (dates `15.07.2025`, fewer broken tokens). Dominant failure mode for EasyOCR on these samples: **CLAHE/denoise → CRAFT over-detect**, not dictionary alone.
2. **Lang:** `ru+en` helps Latin tokens (`e-mail`, `Invoice Packing List`, some `№`) and lowers low-conf % on `nakladnaya`; Cyrillic body stays similar. Recommended EasyOCR lang when Latin appears: `ru+en`.
3. **Confidence alone is not quality:** EasyOCR often reports higher mean conf while still emitting glyph junk (`кСеверТоргж`, `Унизерсальный`). Low-conf re-OCR still useful for borderline tokens.
4. **Tesseract** stays stronger on dense tables / office punctuation on real photos (manual QC); synthetic A/B is complementary evidence for EasyOCR pipeline knobs.

## Pipeline decision from A/B

| Dominates | Change |
|---|---|
| Prep over-fragments EasyOCR | **Done:** lighter EasyOCR preprocess (geometry + gray + deskew; no median/CLAHE). |
| Junk only on low-conf tokens | **Done:** MVP padded / digits allowlist re-OCR in `read_lines`. |
| Ensemble / fine-tune | Later sprints. |

## Related

- Helper: `python docs/_qc_engine_ab.py` (offline, writes `qc-engine-ab-summary.md`).
- Layout work: `plain` / `table`, residual skew, postprocess lookalikes — separate from this engine gap.
- Ensemble still not implemented (README).
