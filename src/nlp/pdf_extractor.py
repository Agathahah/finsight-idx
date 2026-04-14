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

# Heading detection: short lines in title case or ALL CAPS
HEADING_REGEX = re.compile(
    r"^(?:[A-Z][A-Z\s\d]{3,60}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,6})$"
)


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
    ) -> None:
        """
        Args:
            min_section_chars:    Minimum characters for a section to be kept.
            max_pages_per_chunk:  Max pages grouped into one section if no
                                  heading is detected.
        """
        self.min_section_chars = min_section_chars
        self.max_pages_per_chunk = max_pages_per_chunk

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

        pages = self._extract_pages(path)
        if not any(p.text.strip() for p in pages):
            raise ValueError(
                f"No extractable text found in {path.name}. "
                "File may be scanned/image-based — OCR required."
            )

        raw_text = "\n\n".join(p.text for p in pages if p.text.strip())
        sections = self._detect_sections(pages)

        logger.success(
            "Extraction complete | pages={} | sections={} | chars={}",
            len(pages),
            len(sections),
            len(raw_text),
        )

        return ExtractedDocument(
            file_path=str(path),
            total_pages=len(pages),
            pages=pages,
            sections=sections,
            raw_text=raw_text,
            metadata={"filename": path.name, "stem": path.stem},
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _extract_pages(self, path: Path) -> list[PageContent]:
        """Extract text and tables from each page."""
        pages: list[PageContent] = []

        with pdfplumber.open(path) as pdf:
            logger.debug("Total pages in PDF: {}", len(pdf.pages))

            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                text = self._clean_text(raw)

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
                    text="\n\n".join(current_texts),
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
                        text="\n\n".join(current_texts),
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
                text="\n\n".join(current_texts),
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
        clean = text.strip()
        if len(clean) < self.min_section_chars:
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
