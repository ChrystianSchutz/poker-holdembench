"""Render docs/paper.md to docs/holdembench-paper.pdf with a local Chrome or Edge.

uv run python scripts/paper_pdf.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "paper.md"
OUT = ROOT / "docs" / "holdembench-paper.pdf"

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm; }
body { font: 10.4pt/1.5 Georgia, "Times New Roman", serif; color: #111; }
h1 { font: 700 19pt/1.2 "Segoe UI", Arial, sans-serif; margin: 0 0 6pt; }
h2 { font: 700 13pt/1.3 "Segoe UI", Arial, sans-serif; margin: 18pt 0 6pt; }
h3 { font: 700 11pt/1.3 "Segoe UI", Arial, sans-serif; margin: 14pt 0 4pt; }
p, li { text-align: justify; hyphens: auto; }
code { font: 9pt Consolas, monospace; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font: 8pt/1.35 "Segoe UI", Arial, sans-serif; page-break-inside: avoid; }
th, td { padding: 2.5pt 5pt; border-bottom: .5pt solid #ccc; text-align: left; }
th { border-bottom: 1pt solid #111; }
td:not(:nth-child(-n+2)), th:not(:nth-child(-n+2)) { text-align: right; }
img { display: block; max-width: 78%; max-height: 105mm; margin: 10pt auto 2pt; }
p > em:only-child { display: block; font-size: 9pt; color: #444; }
a { color: #14448a; text-decoration: none; }
"""


def chrome() -> str:
    for p in (
        os.environ.get("CHROME"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or shutil.which("chromium"),
    ):
        if p and Path(p).exists():
            return p
    raise SystemExit("No Chrome found; set CHROME=/path/to/chrome")


def main() -> None:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables"])
    html = (
        f'<!doctype html><meta charset="utf-8"><title>HoldemBench</title><style>{CSS}</style>{body}'
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "paper.html"
        shutil.copytree(ROOT / "docs" / "figures", Path(tmp) / "figures")
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                chrome(),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={OUT}",
                page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
