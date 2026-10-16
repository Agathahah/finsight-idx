"""Print the text the indexer sees on one PDF page, to label golden questions.

    python scripts/show_page.py data/raw/bbca_laporan_tahunan_2023.pdf 117
    python scripts/show_page.py data/raw/bbca_laporan_tahunan_2023.pdf 117 --raw   # before header/footer removal

Page numbers are PDF page numbers (what Preview shows), the same numbers the
index stores. They are not the numbers printed in the report's footer.
"""

import argparse
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.nlp.pdf_extractor import PDFExtractor, find_boilerplate, strip_boilerplate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("page", type=int)
    ap.add_argument("--raw", action="store_true", help="show text before header/footer removal")
    args = ap.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        if not 1 <= args.page <= len(pdf.pages):
            sys.exit(f"page must be between 1 and {len(pdf.pages)}")
        text = PDFExtractor._clean_text(pdf.pages[args.page - 1].extract_text() or "")
        if not args.raw:
            # Boilerplate is learned from a sample of pages around the target (fast).
            lo, hi = max(0, args.page - 40), min(len(pdf.pages), args.page + 40)
            sample = [PDFExtractor._clean_text(p.extract_text() or "") for p in pdf.pages[lo:hi]]
            text, _ = strip_boilerplate(text, find_boilerplate(sample))
    print(f"--- PDF page {args.page} ({len(text)} chars) ---")
    print(text or "(no text layer: scanned page, needs OCR)")


if __name__ == "__main__":
    main()
