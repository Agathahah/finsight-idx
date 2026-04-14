"""
FinSight IDX — Document Summarizer
=====================================
Orchestrates PDF extraction → chunking → Claude API summarization
→ structured executive summary output.

Pipeline:
    PDF file
        → PDFExtractor   (extract pages + detect sections)
        → chunk_sections (split large sections into API-safe chunks)
        → Claude API     (summarize each chunk)
        → aggregate      (merge into structured ExecutiveSummary)
        → JSON output
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.api.client import FinancialTask, analyze_financial_text
from src.nlp.pdf_extractor import DocumentSection, ExtractedDocument, PDFExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Max characters per Claude API call (well within context window,
# conservative to keep cost low during development)
MAX_CHUNK_CHARS = 6_000

# Sections to skip (boilerplate content)
SKIP_SECTION_TYPES = set()  # extend if needed: {"governance"}


# ---------------------------------------------------------------------------
# Output Data Models
# ---------------------------------------------------------------------------

@dataclass
class SectionSummary:
    """Summary of a single document section."""
    section_title: str
    section_type: str
    page_range: str
    summary: str
    char_count_original: int


@dataclass
class ExecutiveSummary:
    """
    Structured executive summary of an IDX annual report.
    Designed to be serialized to JSON for downstream processing.
    """
    document_name: str
    processed_at: str
    total_pages: int
    total_sections: int

    # Core output fields
    financial_highlights: str = ""
    risk_factors: str = ""
    outlook_and_strategy: str = ""
    general_overview: str = ""

    # Section-level summaries for traceability
    section_summaries: list[SectionSummary] = field(default_factory=list)

    # Token usage tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> dict:
        """Serialize to plain dict (JSON-compatible)."""
        d = asdict(self)
        d["section_summaries"] = [asdict(s) for s in self.section_summaries]
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class DocumentSummarizer:
    """
    End-to-end pipeline: PDF → structured ExecutiveSummary.

    Usage:
        summarizer = DocumentSummarizer()
        summary = summarizer.summarize("data/raw/laporan_bca_2023.pdf")
        print(summary.financial_highlights)
        summary_json = summary.to_json()
    """

    def __init__(
        self,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        """
        Args:
            max_chunk_chars: Max characters per Claude API call.
            output_dir:      Directory to save JSON output. None = no auto-save.
        """
        self.max_chunk_chars = max_chunk_chars
        self.output_dir = Path(output_dir) if output_dir else None
        self.extractor = PDFExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, pdf_path: str | Path) -> ExecutiveSummary:
        """
        Run the full summarization pipeline on an IDX annual report PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExecutiveSummary with structured financial analysis.

        Raises:
            FileNotFoundError: If PDF does not exist.
            ValueError:        If PDF has no extractable text.
        """
        path = Path(pdf_path)
        logger.info("Starting summarization pipeline | file={}", path.name)

        # Step 1: Extract PDF
        doc = self.extractor.extract(path)

        # Step 2: Initialize output
        summary = ExecutiveSummary(
            document_name=path.stem,
            processed_at=datetime.now().isoformat(),
            total_pages=doc.total_pages,
            total_sections=len(doc.sections),
        )

        # Step 3: Summarize each section
        financial_parts: list[str] = []
        risk_parts: list[str] = []
        outlook_parts: list[str] = []
        general_parts: list[str] = []

        for section in doc.sections:
            if section.section_type in SKIP_SECTION_TYPES:
                logger.debug("Skipping section: {}", section.title)
                continue

            section_summary = self._summarize_section(section)
            summary.section_summaries.append(section_summary)
            summary.total_input_tokens += 0   # updated in _call_claude
            summary.total_output_tokens += 0

            # Route to the right bucket
            if section.section_type == "financial":
                financial_parts.append(section_summary.summary)
            elif section.section_type == "risk":
                risk_parts.append(section_summary.summary)
            elif section.section_type == "outlook":
                outlook_parts.append(section_summary.summary)
            else:
                general_parts.append(section_summary.summary)

        # Step 4: Aggregate per-section summaries into final fields
        summary.financial_highlights = self._aggregate(
            financial_parts, "financial highlights"
        )
        summary.risk_factors = self._aggregate(risk_parts, "risk factors")
        summary.outlook_and_strategy = self._aggregate(
            outlook_parts, "outlook and strategy"
        )
        summary.general_overview = self._aggregate(
            general_parts[:3], "general overview"  # limit to first 3 chunks
        )

        logger.success(
            "Summarization complete | sections={} | in_tokens={} | out_tokens={}",
            len(summary.section_summaries),
            summary.total_input_tokens,
            summary.total_output_tokens,
        )

        # Step 5: Optional auto-save
        if self.output_dir:
            self._save(summary, path.stem)

        return summary

    def summarize_text(
        self,
        text: str,
        document_name: str = "inline_text",
    ) -> ExecutiveSummary:
        """
        Summarize raw text directly (without PDF extraction).
        Useful for testing or when text is already extracted.

        Args:
            text:          Raw financial text to summarize.
            document_name: Label for the output.

        Returns:
            ExecutiveSummary with general_overview populated.
        """
        logger.info("Summarizing raw text | chars={}", len(text))

        summary = ExecutiveSummary(
            document_name=document_name,
            processed_at=datetime.now().isoformat(),
            total_pages=0,
            total_sections=1,
        )

        chunks = self._chunk_text(text)
        parts: list[str] = []

        for i, chunk in enumerate(chunks):
            logger.debug("Processing chunk {}/{}", i + 1, len(chunks))
            result = analyze_financial_text(chunk, task=FinancialTask.SUMMARIZE)
            parts.append(result["result"])
            summary.total_input_tokens += result["usage"]["input_tokens"]
            summary.total_output_tokens += result["usage"]["output_tokens"]

        summary.general_overview = self._aggregate(parts, "overview")
        return summary

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _summarize_section(self, section: DocumentSection) -> SectionSummary:
        """Summarize a single DocumentSection, chunking if necessary."""
        chunks = self._chunk_text(section.text)
        chunk_summaries: list[str] = []

        task = self._section_type_to_task(section.section_type)

        for i, chunk in enumerate(chunks):
            logger.debug(
                "Summarizing section='{}' chunk={}/{} chars={}",
                section.title[:40],
                i + 1,
                len(chunks),
                len(chunk),
            )
            result = analyze_financial_text(chunk, task=task)
            chunk_summaries.append(result["result"])

        # If multiple chunks, do a second-pass merge
        if len(chunk_summaries) > 1:
            merged = "\n\n".join(chunk_summaries)
            final_result = analyze_financial_text(
                merged[:self.max_chunk_chars],
                task=FinancialTask.SUMMARIZE,
            )
            final_text = final_result["result"]
        else:
            final_text = chunk_summaries[0] if chunk_summaries else ""

        return SectionSummary(
            section_title=section.title,
            section_type=section.section_type,
            page_range=f"{section.start_page}–{section.end_page}",
            summary=final_text,
            char_count_original=section.char_count,
        )

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks ≤ max_chunk_chars.
        Splits on paragraph boundaries to preserve context.
        """
        if len(text) <= self.max_chunk_chars:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.max_chunk_chars and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_len = para_len
            else:
                current.append(para)
                current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        logger.debug(
            "Chunked {} chars → {} chunks", len(text), len(chunks)
        )
        return chunks

    def _aggregate(self, parts: list[str], label: str) -> str:
        """
        Merge multiple section summaries into one coherent paragraph.
        If only one part, returns it directly.
        """
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        combined = "\n\n---\n\n".join(parts)
        if len(combined) > self.max_chunk_chars:
            combined = combined[: self.max_chunk_chars]

        logger.debug("Aggregating {} parts for '{}'", len(parts), label)
        result = analyze_financial_text(combined, task=FinancialTask.SUMMARIZE)
        return result["result"]

    @staticmethod
    def _section_type_to_task(section_type: str) -> FinancialTask:
        """Map section type to the most appropriate FinancialTask."""
        mapping = {
            "financial": FinancialTask.KEY_METRICS,
            "risk": FinancialTask.RISK_FACTORS,
            "outlook": FinancialTask.SUMMARIZE,
            "general": FinancialTask.SUMMARIZE,
            "governance": FinancialTask.SUMMARIZE,
        }
        return mapping.get(section_type, FinancialTask.SUMMARIZE)

    def _save(self, summary: ExecutiveSummary, stem: str) -> None:
        """Save ExecutiveSummary as JSON to output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{stem}_summary_{timestamp}.json"
        output_path.write_text(
            summary.to_json(), encoding="utf-8"
        )
        logger.info("Summary saved → {}", output_path)
