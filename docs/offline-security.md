# Offline / network security

Policy: **user photos and OCR text never leave the machine**. Network is allowed only for explicit setup actions the user starts.

## Path matrix

| Path | Network | User data |
|---|---|---|
| CLI / GUI recognition (`ocr.py`, worker) | No (`download_enabled=False` + [`offline_guard`](../offline_guard.py)) | Stays local |
| Spellcheck / preprocess | No | Local |
| GUI **Install libraries** | Yes: `pip` + `https://download.pytorch.org/whl/cpu` | Packages only |
| GUI **Download models** / `--download-models` | Yes: EasyOCR model CDN via EasyOCR | Weights only |
| GUI **Install Tesseract** | Yes: allowlist below | Binary / tessdata only |

## Allowlist (setup only)

| Host / URL pattern | Used by |
|---|---|
| `download.pytorch.org` | GUI library install |
| EasyOCR download endpoints (via `easyocr.Reader(download_enabled=True)`) | `--download-models` |
| `digi.bib.uni-mannheim.de/tesseract/` | Tesseract installer |
| `github.com/tesseract-ocr/...` | Tesseract installer fallback + tessdata |
| `www.7-zip.org/a/7zr.exe` | Extractor for the NSIS installer |

Recognition and the long-lived `--worker` process must not open these URLs. [`offline_guard.offline_session`](../offline_guard.py) blocks `urllib.request.urlopen` during warm + recognize.

## Rules

1. Never upload images, paths, or recognized text.
2. Never download models/binaries during recognition; fail with a clear local error instead.
3. Setup buttons / `--download-models` are the only intentional online steps.
4. After setup, the app can run with Wi‑Fi off.

## How to verify

```bat
python -m unittest tests.test_smoke.OfflineGuardTests -v
```

Manual: install libs/models/Tesseract once, disconnect the network, run GUI recognition on a sample photo.