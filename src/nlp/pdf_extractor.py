"""
FinSight IDX — PDF Extractor
=============================
Extracts structured text from IDX annual report PDFs using pdfplumber.

Responsibilities:
- Page-level text extraction with metadata
- Section detection via heading heuristics
- Table extraction (for financial statements)
- Clean text normalization for downstream NLP
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from loguru import logger


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class PageContent:
    """Extracted content from a single PDF page."""
    page_number: int
    text: str
    tables: list[list] = field(default_factory=list)
    char_count: int = 0
    source: str = "text"  # "text" (PDF text layer), "ocr", or "empty"

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class DocumentSection:
    """A logical section detected within the document."""
    title: str
    start_page: int
    end_page: int
    text: str
    section_type: str = "general"  # financial, risk, outlook, general

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def page_span(self) -> int:
        return self.end_page - self.start_page + 1


@dataclass
class ExtractedDocument:
    """Full extraction result from a PDF file."""
    file_path: str
    total_pages: int
    pages: list[PageContent]
    sections: list[DocumentSection]
    raw_text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Section Detection Patterns
# ---------------------------------------------------------------------------

# Common section headings in IDX annual reports (Indonesian + English)
SECTION_PATTERNS: dict[str, list[str]] = {
    "financial": [
        r"(?i)ikhtisar\s+keuangan",
        r"(?i)kinerja\s+keuangan",
        r"(?i)laporan\s+keuangan",
        r"(?i)financial\s+highlight",
        r"(?i)financial\s+performance",
        r"(?i)pendapatan",
        r"(?i)laba\s+rugi",
    ],
    "risk": [
        r"(?i)faktor\s+risiko",
        r"(?i)manajemen\s+risiko",
        r"(?i)risk\s+factor",
        r"(?i)risk\s+management",
        r"(?i)tantangan",
    ],
    "outlook": [
        r"(?i)prospek",
        r"(?i)outlook",
        r"(?i)strategi",
        r"(?i)rencana\s+ke\s+depan",
        r"(?i)forward\s+looking",
        r"(?i)target\s+\d{4}",
    ],
    "governance": [
        r"(?i)tata\s+kelola",
        r"(?i)good\s+corporate\s+governance",
        r"(?i)dewan\s+komisaris",
        r"(?i)direksi",
    ],
}

# Separator between pages inside a section. Lets the indexer map each chunk
# back to the page it came from (used for citations).
PAGE_BREAK = "\f"

# Heading detection: short lines in title case or ALL CAPS
HEADING_REGEX = re.compile(
    r"^(?:[A-Z][A-Z\s\d]{3,60}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6})$"
)

# Bump when extraction output changes in a way that requires re-indexing.
EXTRACTOR_VERSION = 2

# Repeated lines that carry meaning (units of the numbers in a table) are kept
# even when they appear on many pages.
KEEP_REPEATED = re.compile(r"(?i)jutaan|miliar|ribuan|triliun|rupiah|million|billion")
PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def _line_key(line: str) -> str:
    """Normalise a line so running headers match across pages.

    Digits become '#', and all whitespace is removed because pdfplumber splits
    ligatures inconsistently ("Profi l" vs "Profil", "Tahunan 2023PT").
    """
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", line.lower()))


def _edge_lines(text: str, edge: int) -> list[str]:
    lines = [l for l in text.split("\n") if l.strip()]
    return lines if len(lines) <= 2 * edge else lines[:edge] + lines[-edge:]


def find_boilerplate(page_texts: list[str], min_ratio: float = 0.25,
                     max_len: int = 160, edge: int = 3) -> set[str]:
    """Return keys of lines that repeat at the top or bottom of many pages.

    These are running headers, footers and navigation bars. They add no
    information to a chunk but pull its embedding towards the same text on
    hundreds of other pages. Only the first and last ``edge`` lines of a page
    are considered, so a body line that differs only in its numbers (a table
    row, a recurring sentence) is never counted.
    """
    n = len(page_texts)
    if n < 10:
        return set()
    counts: dict[str, int] = {}
    for text in page_texts:
        for k in {_line_key(l) for l in _edge_lines(text, edge)}:
            counts[k] = counts.get(k, 0) + 1
    return {
        k for k, c in counts.items()
        if c / n >= min_ratio and len(k) <= max_len
        and k.strip("#") and not KEEP_REPEATED.search(k)
    }


def strip_boilerplate(text: str, boilerplate: set[str], edge: int = 3) -> tuple[str, int]:
    """Remove boilerplate lines near the page edges, plus the page-number line."""
    lines = [l for l in text.split("\n") if l.strip()]
    n = len(lines)
    kept = [l for i, l in enumerate(lines)
            if not ((i < edge or i >= n - edge) and _line_key(l) in boilerplate)]
    # A bare number as the first or last line is the page number. Remove at
    # most one at each end: the lines before it can be table values.
    if kept and PAGE_NUMBER_LINE.match(kept[0]):
        kept.pop(0)
    if kept and PAGE_NUMBER_LINE.match(kept[-1]):
        kept.pop()
    return "\n".join(kept).strip(), n - len(kept)


# ---------------------------------------------------------------------------
# OCR (optional: needs the tesseract binary and `pip install pytesseract`)
# ---------------------------------------------------------------------------

def ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_languages(wanted: str) -> str:
    """Use the requested tesseract languages that are installed; fall back to eng."""
    import pytesseract
    try:
        installed = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng"
    langs = [l for l in wanted.split("+") if l in installed]
    return "+".join(langs) or "eng"


def ocr_page_image(image, lang: str = "ind+eng") -> str:
    """OCR a PIL image. Separate function so tests can replace it."""
    import pytesseract
    return pytesseract.image_to_string(image, lang=_ocr_languages(lang))


def looks_like_prose(text: str, min_chars: int = 200, min_alpha: float = 0.6) -> bool:
    """Reject OCR output from photos and decorative pages (short or mostly symbols)."""
    stripped = re.sub(r"\s", "", text)
    if len(stripped) < min_chars:
        return False
    return sum(c.isalpha() for c in stripped) / len(stripped) >= min_alpha


# ---------------------------------------------------------------------------
# Extractor Class
# ---------------------------------------------------------------------------

class PDFExtractor:
    """
    Extracts and structures text from IDX annual report PDFs.

    Usage:
        extractor = PDFExtractor()
        doc = extractor.extract("data/raw/laporan_bca_2023.pdf")
        for section in doc.sections:
            print(section.title, section.section_type)
    """

    def __init__(
        self,
        min_section_chars: int = 200,
        max_pages_per_chunk: int = 5,
        strip_repeated_lines: bool = True,
        boilerplate_min_ratio: float = 0.25,
        ocr: Optional[str] = None,
        ocr_lang: Optional[str] = None,
        ocr_min_chars: int = 50,
        ocr_resolution: int = 300,
    ) -> None:
        """
        Args:
            min_section_chars:     Minimum characters for a section to be kept.
            max_pages_per_chunk:   Max pages grouped into one section if no
                                   heading is detected.
            strip_repeated_lines:  Remove running headers/footers before sectioning.
            boilerplate_min_ratio: Share of pages a line must appear on to count
                                   as a running header/footer.
            ocr:                   "auto" (OCR text-less pages if tesseract is
                                   installed), "off", or "required" (fail if it
                                   is not). Default: env FINSIGHT_OCR or "auto".
            ocr_lang:              Tesseract languages. Default: env OCR_LANG or "ind+eng".
            ocr_min_chars:         Pages with less text than this are OCR candidates.
            ocr_resolution:        DPI used to render a page for OCR.
        """
        self.min_section_chars = min_section_chars
        self.max_pages_per_chunk = max_pages_per_chunk
        self.strip_repeated_lines = strip_repeated_lines
        self.boilerplate_min_ratio = boilerplate_min_ratio
        self.ocr_mode = (ocr or os.getenv("FINSIGHT_OCR", "auto")).lower()
        if self.ocr_mode not in {"auto", "off", "required"}:
            raise ValueError("ocr must be 'auto', 'off' or 'required'")
        self.ocr_lang = ocr_lang or os.getenv("OCR_LANG", "ind+eng")
        self.ocr_min_chars = ocr_min_chars
        self.ocr_resolution = ocr_resolution

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """
        Extract full content from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractedDocument with pages, sections, and raw text.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ValueError:        If the PDF has no extractable text.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info("Extracting PDF | file={} ", path.name)

        use_ocr = self._resolve_ocr()
        pages = self._extract_pages(path, use_ocr=use_ocr)
        if not any(p.text.strip() for p in pages):
            raise ValueError(
                f"No extractable text found in {path.name}. "
                "File may be scanned/image-based: install tesseract and set FINSIGHT_OCR=required."
            )

        removed_lines = 0
        boilerplate: set[str] = set()
        if self.strip_repeated_lines:
            boilerplate = find_boilerplate([p.text for p in pages], self.boilerplate_min_ratio)
            for p in pages:
                p.text, n = strip_boilerplate(p.text, boilerplate)
                p.char_count = len(p.text)
                removed_lines += n

        raw_text = "\n\n".join(p.text for p in pages if p.text.strip())
        sections = self._detect_sections(pages)

        pages_ocr = [p.page_number for p in pages if p.source == "ocr"]
        pages_empty = [p.page_number for p in pages if not p.text.strip()]
        logger.success(
            "Extraction complete | pages={} | sections={} | chars={} | ocr_pages={} | "
            "empty_pages={} | boilerplate_lines_removed={}",
            len(pages), len(sections), len(raw_text),
            len(pages_ocr), len(pages_empty), removed_lines,
        )
        if pages_empty:
            logger.warning("Pages with no text (not searchable): {}", pages_empty)

        return ExtractedDocument(
            file_path=str(path),
            total_pages=len(pages),
            pages=pages,
            sections=sections,
            raw_text=raw_text,
            metadata={
                "filename": path.name,
                "stem": path.stem,
                "extractor_version": EXTRACTOR_VERSION,
                "ocr_mode": self.ocr_mode,
                "ocr_used": use_ocr,
                "pages_ocr": pages_ocr,
                "pages_empty": pages_empty,
                "boilerplate_patterns": len(boilerplate),
                "boilerplate_lines_removed": removed_lines,
            },
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _resolve_ocr(self) -> bool:
        if self.ocr_mode == "off":
            return False
        available = ocr_available()
        if self.ocr_mode == "required" and not available:
            raise RuntimeError(
                "FINSIGHT_OCR=required but tesseract is not available. "
                "macOS: brew install tesseract tesseract-lang && pip install pytesseract"
            )
        if not available:
            logger.warning("OCR off: tesseract/pytesseract not installed; scanned pages stay unsearchable")
        return available

    def _ocr_page(self, page, page_number: int) -> Optional[str]:
        """OCR one pdfplumber page; return cleaned text or None if it is not prose."""
        try:
            image = page.to_image(resolution=self.ocr_resolution).original
            text = self._clean_text(ocr_page_image(image, self.ocr_lang))
        except Exception as exc:
            logger.warning("OCR failed on page {}: {}", page_number, exc)
            return None
        return text if looks_like_prose(text) else None

    def _extract_pages(self, path: Path, use_ocr: bool = False) -> list[PageContent]:
        """Extract text and tables from each page; OCR pages that have no text layer."""
        pages: list[PageContent] = []

        with pdfplumber.open(path) as pdf:
            logger.debug("Total pages in PDF: {}", len(pdf.pages))

            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                text = self._clean_text(raw)
                source = "text" if text.strip() else "empty"

                if use_ocr and len(text) < self.ocr_min_chars and page.images:
                    ocr_text = self._ocr_page(page, i)
                    if ocr_text:
                        text, source = ocr_text, "ocr"
                        logger.info("OCR page {} | chars={}", i, len(text))

                # Extract tables (financial statements)
                tables: list[list] = []
                try:
                    extracted = page.extract_tables()
                    if extracted:
                        tables = extracted
                except Exception as exc:
                    logger.debug("Table extraction failed on page {}: {}", i, exc)

                pages.append(PageContent(
                    page_number=i,
                    text=text,
                    tables=tables,
                    source=source,
                ))

                if i % 20 == 0:
                    logger.debug("Processed {} / {} pages", i, len(pdf.pages))

        return pages

    def _detect_sections(self, pages: list[PageContent]) -> list[DocumentSection]:
        """
        Detect logical sections by scanning for heading-like lines.
        Falls back to fixed-size page windows if no headings found.
        """
        sections: list[DocumentSection] = []
        current_title = "Introduction"
        current_start = 1
        current_texts: list[str] = []

        for page in pages:
            heading = self._find_heading(page.text)

            if heading and current_texts:
                # Save previous section
                section = self._build_section(
                    title=current_title,
                    start_page=current_start,
                    end_page=page.page_number - 1,
                    text=PAGE_BREAK.join(current_texts),
                )
                if section:
                    sections.append(section)

                current_title = heading
                current_start = page.page_number
                current_texts = [page.text]
            else:
                current_texts.append(page.text)

                # Force-split if window too large
                if len(current_texts) >= self.max_pages_per_chunk:
                    section = self._build_section(
                        title=current_title,
                        start_page=current_start,
                        end_page=page.page_number,
                        text=PAGE_BREAK.join(current_texts),
                    )
                    if section:
                        sections.append(section)
                    current_start = page.page_number + 1
                    current_texts = []

        # Flush last section
        if current_texts:
            section = self._build_section(
                title=current_title,
                start_page=current_start,
                end_page=pages[-1].page_number if pages else current_start,
                text=PAGE_BREAK.join(current_texts),
            )
            if section:
                sections.append(section)

        logger.debug("Detected {} sections", len(sections))
        return sections

    def _build_section(
        self,
        title: str,
        start_page: int,
        end_page: int,
        text: str,
    ) -> Optional[DocumentSection]:
        """Build a DocumentSection, returning None if text is too short."""
        # Strip spaces/newlines only: a leading PAGE_BREAK (empty first page)
        # must stay, or every page number in the section shifts by one.
        clean = text.strip(" \n\t\r")
        if len(clean.replace(PAGE_BREAK, "")) < self.min_section_chars:
            return None

        section_type = self._classify_section(title + " " + clean[:200])
        return DocumentSection(
            title=title,
            start_page=start_page,
            end_page=end_page,
            text=clean,
            section_type=section_type,
        )

    def _find_heading(self, text: str) -> Optional[str]:
        """
        Detect a section heading in the first few lines of a page.
        Returns the heading string or None.
        """
        if not text:
            return None

        lines = text.strip().split("\n")[:5]  # Only check first 5 lines
        for line in lines:
            line = line.strip()
            if 4 <= len(line) <= 80 and HEADING_REGEX.match(line):
                return line
        return None

    def _classify_section(self, text: str) -> str:
        """Classify section type based on keyword patterns."""
        for section_type, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return section_type
        return "general"

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize extracted text: remove excessive whitespace and artifacts."""
        if not text:
            return ""
        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u0400-\u04FF\n]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)
        # Collapse more than 2 consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
