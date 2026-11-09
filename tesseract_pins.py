"""Pinned Tesseract download constants (supply-chain freeze).

Do NOT fetch "latest" at runtime. When upstream releases a new Windows build:

1. Open https://github.com/tesseract-ocr/tesseract/releases
2. Copy the new UB-Mannheim ``tesseract-ocr-w64-setup-*.exe`` URL into
   ``TESSERACT_SETUP_URL`` / version fields below.
3. Recompute SHA-256 of that exact file and set ``TESSERACT_SETUP_SHA256``.
4. If tessdata changed, update ``TESSDATA_*`` URL + SHA-256 pairs the same way.
5. Mirror the table in README.md (Offline / Tesseract download sources).

Trust model: official ``github.com/tesseract-ocr`` hosts only + pinned hashes.
"""

from __future__ import annotations

# --- Installer (UB-Mannheim NSIS, published on the tesseract-ocr GitHub release) ---
TESSERACT_VERSION = "5.5.3.20260724"
TESSERACT_TAG = "5.5.3"
TESSERACT_SETUP_NAME = "tesseract-ocr-w64-setup-5.5.3.20260724.exe"
TESSERACT_RELEASE_PAGE = (
    "https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.3"
)
TESSERACT_SETUP_URL = (
    "https://github.com/tesseract-ocr/tesseract/releases/download/"
    "5.5.3/tesseract-ocr-w64-setup-5.5.3.20260724.exe"
)
TESSERACT_SETUP_URLS = (TESSERACT_SETUP_URL,)
TESSERACT_SETUP_SHA256 = (
    "bee9e3434bd94fd65387d9be28cd467a41f61b1275383b55b0f59a1331270ae4"
)
TESSERACT_PUBLISHER = "UB-Mannheim / tesseract-ocr (GitHub)"

# --- Language packs (official tessdata repo) ---
TESSDATA_ENG_URL = (
    "https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata"
)
TESSDATA_ENG_SHA256 = (
    "daa0c97d651c19fba3b25e81317cd697e9908c8208090c94c3905381c23fc047"
)
TESSDATA_RUS_URL = (
    "https://github.com/tesseract-ocr/tessdata/raw/main/rus.traineddata"
)
TESSDATA_RUS_SHA256 = (
    "681be2c2bead1bc7bd235df88c44e8e60ae73ae866840c0ad4e3b4c247bd37c2"
)

# name -> (url, sha256)
TESSDATA_FILES: dict[str, tuple[str, str]] = {
    "eng.traineddata": (TESSDATA_ENG_URL, TESSDATA_ENG_SHA256),
    "rus.traineddata": (TESSDATA_RUS_URL, TESSDATA_RUS_SHA256),
}

REQUIRED_TESSERACT_LANGS = ("eng", "rus")
