# Offline / network security

Policy: **user photos and OCR text never leave the machine**. Network is allowed only for explicit setup actions the user starts.

## Path matrix

| Path | Network | User data |
|---|---|---|
| CLI / GUI recognition (`ocr.py`, worker) | No (`download_enabled=False` + [`offline_guard`](../offline_guard.py)) | Stays local |
| Spellcheck / preprocess | No | Local |
| GUI **Install libraries** | Yes: `pip` + `https://download.pytorch.org/whl/cpu` | Packages only |
| GUI **Download models** / `--download-models` | Yes: EasyOCR model CDN via EasyOCR | Weights only |
| GUI **Install Tesseract** | Yes: allowlist below; one UAC prompt | Binary / tessdata only |

## Allowlist (setup only)

| Host / URL pattern | Used by |
|---|---|
| `download.pytorch.org` | GUI library install |
| EasyOCR download endpoints (via `easyocr.Reader(download_enabled=True)`) | `--download-models` |
| `github.com/tesseract-ocr/tesseract/releases/.../5.5.3/...exe` | UB-Mannheim NSIS installer |
| `github.com/tesseract-ocr/tessdata/raw/main/{eng,rus}.traineddata` | Language packs |

Tesseract setup runs the official NSIS installer silently (`/S /D=%LOCALAPPDATA%\\DocOCR\\tesseract`) with elevation — it cannot be unpacked as a plain archive.

Downloaded installer and `eng`/`rus` tessdata use **pinned URLs + SHA-256** from [`tesseract_pins.py`](../tesseract_pins.py) (never “latest”). A mismatch deletes the file and aborts install. See README *Tesseract download sources*.

Before UAC, the GUI shows the official GitHub release URL, publisher (UB-Mannheim / tesseract-ocr), and the expected SHA-256. Windows SmartScreen / UAC may still say **Unknown publisher** — the community NSIS build is often not Authenticode-signed; trust comes from the pinned official URL + hash, not from a Store certificate.

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