"""
FinSight IDX — Agent Orchestrator
===================================
End-to-end analysis pipeline that automates the full FinSight workflow:

    Input: emiten + tahun
        → Step 1: Fetch & summarize annual report (PDF → structured summary)
        → Step 2: Topic modeling on financial news (BERTopic)
        → Step 3: Sentiment comparison (news vs report tone)
        → Step 4: Generate final Markdown report

This is the flagship feature of FinSight IDX — combining all modules
into a single automated analysis agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.api.client import get_client
from src.api.tools import (
    FinancialAnalystAgent,
    cari_di_laporan,
    hitung_rasio_keuangan,
)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SentimentComparison:
    """Comparison of news sentiment vs report tone."""
    news_sentiment: str        # Positif / Netral / Negatif
    news_score: float          # 0.0 - 1.0
    report_tone: str           # Optimistis / Netral / Konservatif
    alignment: str             # Selaras / Divergen / Kontradiktif
    divergence_notes: str = ""
    benchmark_notes: str = ""  # comparison to BI press release baseline


@dataclass
class TopicSummary:
    """Summary of topic modeling results."""
    n_topics: int
    top_topics: list[dict]
    dominant_theme: str
    outlier_rate: float


@dataclass
class AnalysisReport:
    """Full analysis report output from the orchestrator."""
    emiten: str
    tahun: int
    generated_at: str

    # Pipeline outputs
    executive_summary: str = ""
    financial_highlights: str = ""
    risk_factors: str = ""
    outlook: str = ""
    topic_summary: Optional[TopicSummary] = None
    sentiment_comparison: Optional[SentimentComparison] = None

    # Final report
    markdown_report: str = ""

    # Metadata
    steps_completed: list[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_seconds: float = 0.0

    def save(self, output_dir: str | Path = "data/processed") -> Path:
        """Save markdown report to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analisis_{self.emiten}_{self.tahun}_{ts}.md"
        path = output_path / filename
        path.write_text(self.markdown_report, encoding="utf-8")
        logger.success("Report saved → {}", path)
        return path


# ---------------------------------------------------------------------------
# Sentiment Analyzer (Claude-based)
# ---------------------------------------------------------------------------

SENTIMENT_SYSTEM_PROMPT = """Anda adalah analis sentimen keuangan yang menganalisis
teks berita keuangan Indonesia. Berikan respons HANYA dalam format JSON:
{
  "sentimen": "Positif|Netral|Negatif",
  "skor": <float 0.0-1.0>,
  "tone_laporan": "Optimistis|Netral|Konservatif",
  "alignment": "Selaras|Divergen|Kontradiktif",
  "catatan": "<penjelasan singkat>"
}"""


def analyze_sentiment_comparison(
    news_texts: list[str],
    report_summary: str,
    emiten: str,
) -> SentimentComparison:
    """
    Compare sentiment of news coverage vs tone of annual report.

    Benchmarks against BI press release communication style
    (formal, data-driven, forward-looking).

    Args:
        news_texts:     List of news article texts about the company.
        report_summary: Summary of the annual report.
        emiten:         Company ticker for context.

    Returns:
        SentimentComparison with alignment analysis.
    """
    if not news_texts:
        return SentimentComparison(
            news_sentiment="Netral",
            news_score=0.5,
            report_tone="Netral",
            alignment="Tidak dapat dinilai — tidak ada berita",
        )

    news_sample = "\n".join(news_texts[:10])[:3000]

    user_message = f"""Analisis sentimen untuk emiten {emiten}:

=== BERITA TERKINI ===
{news_sample}

=== RINGKASAN LAPORAN TAHUNAN ===
{report_summary[:1500]}

=== KONTEKS BENCHMARK ===
Bandingkan dengan gaya komunikasi Bank Indonesia yang cenderung:
- Formal dan data-driven
- Tone hati-hati namun konstruktif
- Forward-looking dengan target terukur

Berikan analisis sentimen dalam format JSON yang diminta."""

    try:
        response = get_client().messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=512,
            temperature=0.1,
            system=SENTIMENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw.strip())

        return SentimentComparison(
            news_sentiment=data.get("sentimen", "Netral"),
            news_score=float(data.get("skor", 0.5)),
            report_tone=data.get("tone_laporan", "Netral"),
            alignment=data.get("alignment", "Tidak dapat dinilai"),
            divergence_notes=data.get("catatan", ""),
            benchmark_notes=(
                "Dibandingkan gaya komunikasi press release BI: "
                "formal, data-driven, konstruktif dengan target terukur."
            ),
        )

    except Exception as exc:
        logger.warning("Sentiment analysis failed: {}", exc)
        return SentimentComparison(
            news_sentiment="Netral",
            news_score=0.5,
            report_tone="Netral",
            alignment="Error dalam analisis",
            divergence_notes=str(exc),
        )


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """Anda adalah analis keuangan senior yang menulis laporan
analisis investasi profesional untuk pasar modal Indonesia.
Tulis laporan dalam Bahasa Indonesia yang formal, terstruktur, dan mudah dibaca.
Sertakan data kuantitatif dan interpretasi kualitatif."""

REPORT_USER_TEMPLATE = """Buat laporan analisis fundamental komprehensif berdasarkan data berikut:

## Data Emiten: {emiten} ({tahun})

### Ringkasan Eksekutif dari Laporan Tahunan:
{executive_summary}

### Highlights Keuangan:
{financial_highlights}

### Faktor Risiko:
{risk_factors}

### Prospek & Strategi:
{outlook}

### Analisis Topik Berita:
{topic_analysis}

### Perbandingan Sentimen:
{sentiment_analysis}

---
Tulis laporan Markdown yang mencakup:
1. **Ikhtisar Perusahaan** — profil singkat dan posisi di industri
2. **Kinerja Keuangan** — analisis angka-angka kunci dengan tren
3. **Analisis Risiko** — identifikasi dan penilaian risiko material
4. **Perspektif Pasar** — sentimen berita vs tone manajemen
5. **Valuasi & Outlook** — penilaian nilai dan prospek
6. **Kesimpulan** — ringkasan dengan poin-poin kunci

Tambahkan disclaimer bahwa laporan ini bukan rekomendasi investasi."""


def generate_markdown_report(
    emiten: str,
    tahun: int,
    executive_summary: str,
    financial_highlights: str,
    risk_factors: str,
    outlook: str,
    topic_summary: Optional[TopicSummary],
    sentiment: Optional[SentimentComparison],
) -> tuple[str, int, int]:
    """
    Generate final Markdown analysis report using Claude.

    Returns:
        Tuple of (markdown_text, input_tokens, output_tokens).
    """
    topic_text = "Tidak tersedia (BERTopic tidak terinstall)"
    if topic_summary:
        top = ", ".join(
            f"{t['label']} ({t['size']} artikel)"
            for t in topic_summary.top_topics[:5]
        )
        topic_text = (
            f"{topic_summary.n_topics} topik teridentifikasi. "
            f"Tema dominan: {topic_summary.dominant_theme}. "
            f"Top topics: {top}"
        )

    sentiment_text = "Tidak tersedia"
    if sentiment:
        sentiment_text = (
            f"Sentimen berita: {sentiment.news_sentiment} "
            f"(skor: {sentiment.news_score:.2f}). "
            f"Tone laporan: {sentiment.report_tone}. "
            f"Alignment: {sentiment.alignment}. "
            f"{sentiment.divergence_notes}"
        )

    user_message = REPORT_USER_TEMPLATE.format(
        emiten=emiten.upper(),
        tahun=tahun,
        executive_summary=executive_summary[:1000] or "Tidak tersedia",
        financial_highlights=financial_highlights[:800] or "Tidak tersedia",
        risk_factors=risk_factors[:800] or "Tidak tersedia",
        outlook=outlook[:600] or "Tidak tersedia",
        topic_analysis=topic_text,
        sentiment_analysis=sentiment_text,
    )

    response = get_client().messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=3000,
        temperature=0.2,
        system=REPORT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return (
        response.content[0].text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

class FinSightOrchestrator:
    """
    End-to-end analysis orchestrator for IDX annual reports.

    Combines PDF extraction, summarization, topic modeling,
    sentiment analysis, and report generation into one pipeline.

    Usage:
        orchestrator = FinSightOrchestrator()
        report = orchestrator.analyze(
            emiten="BBCA",
            tahun=2023,
            pdf_path="data/raw/bbca_laporan_tahunan_2023.pdf",
        )
        print(report.markdown_report)
        report.save("data/processed")
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed",
        rag_persist_dir: str = "data/vectorstore",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.rag_persist_dir = rag_persist_dir

    def analyze(
        self,
        emiten: str,
        tahun: int,
        pdf_path: Optional[str | Path] = None,
        news_texts: Optional[list[str]] = None,
        skip_topic_modeling: bool = False,
        skip_rag: bool = False,
    ) -> AnalysisReport:
        """
        Run full analysis pipeline for an IDX company.

        Args:
            emiten:               Company ticker (e.g., BBCA).
            tahun:                Report year (e.g., 2023).
            pdf_path:             Path to annual report PDF.
            news_texts:           News articles for topic modeling + sentiment.
                                  If None, uses synthetic dataset.
            skip_topic_modeling:  Skip BERTopic (faster, saves time).
            skip_rag:             Skip RAG retrieval (if not indexed).

        Returns:
            AnalysisReport with full analysis and Markdown report.
        """
        start_time = datetime.now()
        emiten = emiten.upper()

        logger.info(
            "Starting FinSight analysis | emiten={} | tahun={}",
            emiten, tahun
        )

        report = AnalysisReport(
            emiten=emiten,
            tahun=tahun,
            generated_at=datetime.now().isoformat(),
        )

        # ── Step 1: Summarize Annual Report ──────────────────────────────
        logger.info("Step 1: Summarizing annual report...")
        try:
            report = self._step_summarize(report, pdf_path)
            report.steps_completed.append("summarize")
        except Exception as exc:
            logger.warning("Step 1 failed: {} — continuing", exc)
            report.executive_summary = f"Summarization gagal: {exc}"

        # ── Step 2: Topic Modeling ────────────────────────────────────────
        if not skip_topic_modeling:
            logger.info("Step 2: Running topic modeling...")
            try:
                texts = news_texts or self._get_sample_news(emiten)
                report = self._step_topic_modeling(report, texts)
                report.steps_completed.append("topic_modeling")
            except Exception as exc:
                logger.warning("Step 2 failed: {} — continuing", exc)
        else:
            logger.info("Step 2: Skipped (skip_topic_modeling=True)")

        # ── Step 3: Sentiment Analysis ────────────────────────────────────
        logger.info("Step 3: Analyzing sentiment...")
        try:
            texts = news_texts or self._get_sample_news(emiten)
            sentiment = analyze_sentiment_comparison(
                news_texts=texts,
                report_summary=report.executive_summary,
                emiten=emiten,
            )
            report.sentiment_comparison = sentiment
            report.total_input_tokens += 200   # approximate
            report.steps_completed.append("sentiment")
        except Exception as exc:
            logger.warning("Step 3 failed: {} — continuing", exc)

        # ── Step 4: RAG Enhancement ───────────────────────────────────────
        if not skip_rag:
            logger.info("Step 4: Enhancing with RAG retrieval...")
            try:
                report = self._step_rag_enhancement(report, emiten, tahun)
                report.steps_completed.append("rag")
            except Exception as exc:
                logger.warning("Step 4 failed (RAG): {} — continuing", exc)
        else:
            logger.info("Step 4: Skipped (skip_rag=True)")

        # ── Step 5: Generate Final Report ────────────────────────────────
        logger.info("Step 5: Generating Markdown report...")
        try:
            md, in_tok, out_tok = generate_markdown_report(
                emiten=emiten,
                tahun=tahun,
                executive_summary=report.executive_summary,
                financial_highlights=report.financial_highlights,
                risk_factors=report.risk_factors,
                outlook=report.outlook,
                topic_summary=report.topic_summary,
                sentiment=report.sentiment_comparison,
            )
            report.markdown_report = md
            report.total_input_tokens += in_tok
            report.total_output_tokens += out_tok
            report.steps_completed.append("report_generation")
        except Exception as exc:
            logger.error("Step 5 failed: {}", exc)
            report.markdown_report = self._fallback_report(report)

        report.duration_seconds = (
            datetime.now() - start_time
        ).total_seconds()

        logger.success(
            "Analysis complete | emiten={} | steps={} | duration={:.1f}s | tokens={}/{}",
            emiten,
            report.steps_completed,
            report.duration_seconds,
            report.total_input_tokens,
            report.total_output_tokens,
        )

        return report

    # ------------------------------------------------------------------
    # Pipeline Steps
    # ------------------------------------------------------------------

    def _step_summarize(
        self,
        report: AnalysisReport,
        pdf_path: Optional[str | Path],
    ) -> AnalysisReport:
        """Step 1: Extract and summarize PDF."""
        from src.nlp.summarizer import DocumentSummarizer

        if not pdf_path:
            pdf_path = self._find_pdf(report.emiten, report.tahun)

        summarizer = DocumentSummarizer(output_dir=str(self.output_dir))
        summary = summarizer.summarize(pdf_path)

        report.executive_summary = summary.general_overview
        report.financial_highlights = summary.financial_highlights
        report.risk_factors = summary.risk_factors
        report.outlook = summary.outlook_and_strategy
        report.total_input_tokens += summary.total_input_tokens
        report.total_output_tokens += summary.total_output_tokens

        return report

    def _step_topic_modeling(
        self,
        report: AnalysisReport,
        texts: list[str],
    ) -> AnalysisReport:
        """Step 2: Run BERTopic on news texts."""
        from src.nlp.topic_modeler import FinancialTopicModeler

        if len(texts) < 10:
            logger.warning("Too few texts for topic modeling: {}", len(texts))
            return report

        modeler = FinancialTopicModeler(min_topic_size=5)
        result = modeler.fit(
            texts,
            model_name=f"{report.emiten}_{report.tahun}_topics",
        )

        top_topics = [
            {
                "label": t.label,
                "size": t.size,
                "words": t.representative_words[:5],
            }
            for t in result.topics[:5]
        ]

        dominant = result.topics[0].label if result.topics else "Tidak teridentifikasi"

        report.topic_summary = TopicSummary(
            n_topics=result.n_topics,
            top_topics=top_topics,
            dominant_theme=dominant,
            outlier_rate=round(
                result.outlier_count / result.n_documents, 3
            ) if result.n_documents > 0 else 0.0,
        )
        return report

    def _step_rag_enhancement(
        self,
        report: AnalysisReport,
        emiten: str,
        tahun: int,
    ) -> AnalysisReport:
        """Step 4: Enhance report with specific RAG queries."""
        queries = [
            f"Berapa laba bersih {emiten} {tahun}?",
            f"Apa strategi utama {emiten} untuk pertumbuhan?",
        ]

        enhancements = []
        for query in queries:
            result = cari_di_laporan(
                query=query,
                emiten=emiten,
                tahun=tahun,
                persist_dir=self.rag_persist_dir,
            )
            if result.get("hasil"):
                top = result["hasil"][0]
                enhancements.append(
                    f"[hal. {top['halaman']}] {top['kutipan'][:200]}"
                )

        if enhancements and not report.financial_highlights:
            report.financial_highlights = "\n\n".join(enhancements)

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_pdf(self, emiten: str, tahun: int) -> Path:
        """Find PDF in data/raw/ by emiten and year."""
        raw_dir = Path("data/raw")
        for pattern in [
            f"*{emiten.lower()}*{tahun}*.pdf",
            f"*{tahun}*{emiten.lower()}*.pdf",
            f"*{emiten.upper()}*{tahun}*.pdf",
        ]:
            candidates = list(raw_dir.glob(pattern))
            if candidates:
                return candidates[0]

        available = [f.name for f in raw_dir.glob("*.pdf")]
        raise FileNotFoundError(
            f"PDF untuk {emiten} {tahun} tidak ditemukan di data/raw/. "
            f"File tersedia: {available}"
        )

    def _get_sample_news(self, emiten: str) -> list[str]:
        """Generate sample news texts if none provided."""
        from src.nlp.news_scraper import generate_sample_dataset
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_sample_dataset(n_articles=100, output_dir=tmpdir)
            import json as _json
            texts = [
                _json.loads(line)["text"]
                for line in path.read_text().strip().split("\n")
            ]
        return texts

    def _fallback_report(self, report: AnalysisReport) -> str:
        """Generate a minimal markdown report from available data."""
        return f"""# Laporan Analisis FinSight IDX
## {report.emiten} — {report.tahun}

*Dihasilkan: {report.generated_at}*

---

## Ringkasan Eksekutif
{report.executive_summary or 'Data tidak tersedia'}

## Highlights Keuangan
{report.financial_highlights or 'Data tidak tersedia'}

## Faktor Risiko
{report.risk_factors or 'Data tidak tersedia'}

## Prospek & Strategi
{report.outlook or 'Data tidak tersedia'}

---
*Disclaimer: Laporan ini dihasilkan secara otomatis oleh FinSight IDX
dan bukan merupakan rekomendasi investasi.*
"""
