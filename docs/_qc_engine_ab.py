"""Offline A/B: EasyOCR vs Tesseract confidence metrics on resources/*.png.

Writes summary to stdout and docs/qc-engine-ab-summary.md.
Does not upload images or touch the network.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "resources"
OUT_MD = ROOT / "docs" / "qc-engine-ab-summary.md"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = str(VENV_PY if VENV_PY.exists() else Path(sys.executable))
FIXTURES = [
    "zayavlenie_otpusk.png",
    "spravka_s_mesta_raboty.png",
    "nakladnaya_tovarnaya.png",
]
LOW_CONF = 0.65
CONF_LINE = re.compile(r"^\d+\.\s+\[(\d+\.\d+)\](\s+\[!\])?\s+(.*)$")


def parse_debug(path: Path) -> dict[str, float | int]:
    if not path.is_file():
        return {"words": 0, "low": 0, "mean_conf": 0.0}
    confs: list[float] = []
    low = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CONF_LINE.match(raw.strip())
        if not match:
            continue
        conf = float(match.group(1))
        confs.append(conf)
        if conf < LOW_CONF or match.group(2):
            low += 1
    words = len(confs)
    mean = sum(confs) / words if words else 0.0
    return {"words": words, "low": low, "mean_conf": mean}


def run_case(
    image: Path,
    *,
    engine: str,
    lang: str,
    preprocess: bool,
    dest_dir: Path,
) -> dict[str, object]:
    stem = f"{image.stem}_{engine}_{lang}_{'prep' if preprocess else 'noprep'}"
    out_txt = dest_dir / f"{stem}.txt"
    cmd = [
        PY,
        str(ROOT / "ocr.py"),
        str(image),
        "-o",
        str(out_txt),
        "--engine",
        engine,
        "--lang",
        lang,
        "--ui",
        "en",
        "--debug",
        "--no-spell",
        "--no-fix-caps",
    ]
    if not preprocess:
        cmd.append("--no-preprocess")
    code = subprocess.call(cmd, cwd=str(ROOT))
    debug_path = out_txt.with_name(f"{out_txt.stem}.debug{out_txt.suffix}")
    stats = parse_debug(debug_path)
    text = out_txt.read_text(encoding="utf-8", errors="replace") if out_txt.exists() else ""
    return {
        "image": image.name,
        "engine": engine,
        "lang": lang,
        "preprocess": preprocess,
        "exit": code,
        "words": stats["words"],
        "low": stats["low"],
        "low_pct": (100.0 * float(stats["low"]) / float(stats["words"]))
        if stats["words"]
        else 0.0,
        "mean_conf": stats["mean_conf"],
        "out": str(out_txt),
        "debug": str(debug_path),
        "preview": " ".join(text.split())[:180],
    }


def main() -> int:
    images = [RESOURCES / name for name in FIXTURES if (RESOURCES / name).is_file()]
    if not images:
        print("No fixtures in resources/", file=sys.stderr)
        return 1

    cases: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="qc_engine_ab_") as tmp:
        dest = Path(tmp)
        for image in images:
            cases.append(
                run_case(
                    image, engine="tesseract", lang="ru", preprocess=True, dest_dir=dest
                )
            )
            cases.append(
                run_case(
                    image, engine="easyocr", lang="ru", preprocess=True, dest_dir=dest
                )
            )
            cases.append(
                run_case(
                    image,
                    engine="easyocr",
                    lang="ru",
                    preprocess=False,
                    dest_dir=dest,
                )
            )
            cases.append(
                run_case(
                    image,
                    engine="easyocr",
                    lang="ru+en",
                    preprocess=True,
                    dest_dir=dest,
                )
            )

        lines = [
            "# EasyOCR vs Tesseract A/B summary",
            "",
            "Generated offline by `docs/_qc_engine_ab.py`.",
            "",
            "| Image | Engine | Lang | Prep | Words | Low conf | Low % | Mean conf | Exit |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in cases:
            lines.append(
                "| {image} | {engine} | {lang} | {prep} | {words} | {low} | {low_pct:.1f} | "
                "{mean:.3f} | {exit} |".format(
                    image=row["image"],
                    engine=row["engine"],
                    lang=row["lang"],
                    prep="yes" if row["preprocess"] else "no",
                    words=row["words"],
                    low=row["low"],
                    low_pct=float(row["low_pct"]),
                    mean=float(row["mean_conf"]),
                    exit=row["exit"],
                )
            )
        lines.append("")
        lines.append("## Previews")
        lines.append("")
        for row in cases:
            lines.append(
                f"- **{row['image']} / {row['engine']} / {row['lang']} / "
                f"{'prep' if row['preprocess'] else 'noprep'}:** {row['preview']}"
            )
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(OUT_MD.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
