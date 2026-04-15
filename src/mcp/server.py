"""
FinSight IDX — MCP Server
==========================
Exposes FinSight capabilities as MCP tools, resources, and prompt templates
using FastMCP. Compatible with Claude Desktop and MCP Inspector.

Tools:
    - summarize_laporan    : Summarize IDX annual report PDF
    - topic_modeling       : Run BERTopic on financial news texts
    - tanya_laporan        : RAG Q&A on indexed documents
    - hitung_rasio         : Compute financial ratios

Resources:
    - finsight://corpus    : List of indexed documents in ChromaDB

Prompt Templates:
    - analisis_fundamental : Full fundamental analysis template
    - deteksi_risiko       : Risk detection template

Run:
    python -m src.mcp.server
    # or
    fastmcp run src/mcp/server.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    FastMCP = None  # type: ignore

# FinSight imports
from src.api.tools import (
    hitung_rasio_keuangan,
    bandingkan_emiten,
    cari_di_laporan,
)

# ---------------------------------------------------------------------------
# Server Instance
# ---------------------------------------------------------------------------

if HAS_FASTMCP:
    mcp = FastMCP(
        name="FinSight IDX",
        instructions=(
            "Financial NLP intelligence server for Indonesian capital markets (IDX). "
            "Provides tools for summarizing annual reports, topic modeling of financial "
            "news, RAG-based Q&A, and financial ratio calculations. "
            "All data is sourced from publicly available IDX documents."
        ),
    )
else:
    mcp = None  # type: ignore

# Configuration from environment
RAG_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/vectorstore")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _tool(f):
    """Decorator: register as MCP tool if FastMCP available, else passthrough."""
    return mcp.tool()(f) if HAS_FASTMCP else f

def _resource(uri):
    def decorator(f):
        return mcp.resource(uri)(f) if HAS_FASTMCP else f
    return decorator

def _prompt():
    def decorator(f):
        return mcp.prompt()(f) if HAS_FASTMCP else f
    return decorator


@_tool
def summarize_laporan(
    emiten: str,
    tahun: int,
    pdf_path: Optional[str] = None,
) -> str:
    """
    Summarize an IDX annual report for a given company and year.

    Extracts key financial highlights, risk factors, and outlook from
    the annual report PDF. Returns a structured JSON summary.

    Args:
        emiten:   Company ticker code (e.g., BBCA, BBRI, BMRI).
        tahun:    Report year (e.g., 2023).
        pdf_path: Optional path to PDF file. If not provided, searches
                  data/raw/ for matching file.

    Returns:
        JSON string with financial_highlights, risk_factors, outlook.
    """
    from src.nlp.summarizer import DocumentSummarizer
    from src.nlp.pdf_extractor import PDFExtractor

    logger.info("MCP: summarize_laporan | emiten={} tahun={}", emiten, tahun)

    # Find PDF if path not provided
    if not pdf_path:
        raw_dir = Path("data/raw")
        candidates = list(raw_dir.glob(f"*{emiten.lower()}*{tahun}*.pdf"))
        if not candidates:
            candidates = list(raw_dir.glob(f"*{tahun}*{emiten.lower()}*.pdf"))
        if not candidates:
            return json.dumps({
                "error": (
                    f"PDF untuk {emiten} {tahun} tidak ditemukan di data/raw/. "
                    f"File yang tersedia: {[f.name for f in raw_dir.glob('*.pdf')]}"
                )
            }, ensure_ascii=False)
        pdf_path = str(candidates[0])

    try:
        summarizer = DocumentSummarizer(
            output_dir="data/processed",
        )
        result = summarizer.summarize(pdf_path)
        return json.dumps({
            "emiten": emiten.upper(),
            "tahun": tahun,
            "document_name": result.document_name,
            "total_pages": result.total_pages,
            "total_sections": result.total_sections,
            "financial_highlights": result.financial_highlights,
            "risk_factors": result.risk_factors,
            "outlook_and_strategy": result.outlook_and_strategy,
            "general_overview": result.general_overview[:500],
            "tokens_used": {
                "input": result.total_input_tokens,
                "output": result.total_output_tokens,
            },
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        logger.error("summarize_laporan failed: {}", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@_tool
def topic_modeling(
    teks_list: list[str],
    model_name: str = "finsight_topics",
    min_topic_size: int = 3,
) -> str:
    """
    Run BERTopic neural topic modeling on a list of financial news texts.

    Identifies dominant topics in Indonesian financial news such as
    interest rates, stock market movements, commodity prices, etc.

    Args:
        teks_list:      List of article/document texts (min 10 required).
        model_name:     Label for the output model.
        min_topic_size: Minimum documents per topic cluster.

    Returns:
        JSON string with discovered topics, representative words, and sizes.
    """
    logger.info("MCP: topic_modeling | n_texts={}", len(teks_list))

    if len(teks_list) < 10:
        return json.dumps({
            "error": f"Minimal 10 teks diperlukan, diberikan {len(teks_list)}."
        }, ensure_ascii=False)

    try:
        from src.nlp.topic_modeler import FinancialTopicModeler
        modeler = FinancialTopicModeler(min_topic_size=min_topic_size)
        result = modeler.fit(teks_list, model_name=model_name)

        return json.dumps({
            "model_name": result.model_name,
            "n_documents": result.n_documents,
            "n_topics": result.n_topics,
            "outlier_count": result.outlier_count,
            "topics": [
                {
                    "id": t.topic_id,
                    "label": t.label,
                    "size": t.size,
                    "top_words": t.representative_words[:5],
                    "probability": t.probability,
                }
                for t in result.topics[:10]  # top 10 topics
            ],
            "summary": result.summary(),
        }, ensure_ascii=False, indent=2)

    except ImportError:
        return json.dumps({
            "error": "BERTopic tidak terinstall. Run: pip install bertopic sentence-transformers"
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("topic_modeling failed: {}", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@_tool
def tanya_laporan(
    query: str,
    emiten: Optional[str] = None,
    tahun: Optional[int] = None,
) -> str:
    """
    Ask a question about indexed IDX annual reports using RAG.

    Retrieves relevant passages from ChromaDB and generates a cited answer
    using Claude. Returns the answer with source citations (page numbers).

    Args:
        query:  Natural language question about the annual report.
        emiten: Optional company filter (e.g., BBCA).
        tahun:  Optional year filter (e.g., 2023).

    Returns:
        JSON string with answer, citations, and source excerpts.
    """
    logger.info("MCP: tanya_laporan | query='{}' emiten={}", query[:60], emiten)

    try:
        from src.rag.qa_chain import FinancialQAChain
        qa = FinancialQAChain(
            persist_dir=RAG_PERSIST_DIR,
            model=CLAUDE_MODEL,
        )
        response = qa.ask(
            question=query,
            company_filter=emiten,
            year_filter=tahun,
        )

        return json.dumps({
            "query": response.question,
            "answer": response.answer,
            "sources": [
                {
                    "citation": src.citation,
                    "halaman": src.page_number,
                    "seksi": src.section_title,
                    "preview": src.text_preview[:200],
                }
                for src in response.sources
            ],
            "retrieved_chunks": response.retrieved_chunks,
            "tokens": {
                "input": response.input_tokens,
                "output": response.output_tokens,
            },
            "latency_ms": response.latency_ms,
        }, ensure_ascii=False, indent=2)

    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
            "hint": "Jalankan RAGIndexer.index_pdf() terlebih dahulu.",
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("tanya_laporan failed: {}", exc)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


@_tool
def hitung_rasio(
    emiten: str,
    harga_saham: Optional[float] = None,
    eps: Optional[float] = None,
    book_value_per_share: Optional[float] = None,
    laba_bersih: Optional[float] = None,
    ekuitas: Optional[float] = None,
    total_hutang: Optional[float] = None,
    total_aset: Optional[float] = None,
    pendapatan: Optional[float] = None,
) -> str:
    """
    Calculate financial ratios for an IDX-listed company.

    Computes PER, PBV, ROE, DER, NPM, ROA with automatic interpretation
    and comparison to Indonesian banking sector benchmarks.

    Args:
        emiten:               Company ticker (e.g., BBCA).
        harga_saham:          Current stock price in IDR.
        eps:                  Earnings per share in IDR.
        book_value_per_share: Book value per share in IDR.
        laba_bersih:          Net profit in billion IDR.
        ekuitas:              Total equity in billion IDR.
        total_hutang:         Total debt in billion IDR.
        total_aset:           Total assets in billion IDR.
        pendapatan:           Total revenue in billion IDR.

    Returns:
        JSON string with computed ratios and interpretations.
    """
    logger.info("MCP: hitung_rasio | emiten={}", emiten)

    result = hitung_rasio_keuangan(
        emiten=emiten,
        harga_saham=harga_saham,
        eps=eps,
        book_value_per_share=book_value_per_share,
        laba_bersih=laba_bersih,
        ekuitas=ekuitas,
        total_hutang=total_hutang,
        total_aset=total_aset,
        pendapatan=pendapatan,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@_resource("finsight://corpus")
def get_corpus_info() -> str:
    """
    List all documents indexed in the FinSight ChromaDB vector store.

    Returns metadata about indexed annual reports available for querying,
    including company names, years, and chunk counts.
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=RAG_PERSIST_DIR)
        collection = client.get_or_create_collection("finsight_documents")

        total_chunks = collection.count()
        if total_chunks == 0:
            return json.dumps({
                "status": "empty",
                "message": "Belum ada dokumen terindeks.",
                "hint": "Jalankan RAGIndexer.index_pdf() untuk mengindeks laporan.",
            }, ensure_ascii=False, indent=2)

        # Get unique documents
        results = collection.get(
            limit=1000,
            include=["metadatas"],
        )

        docs: dict[str, dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "company": meta.get("company", ""),
                    "year": meta.get("year", 0),
                    "source_file": meta.get("source_file", ""),
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return json.dumps({
            "status": "ok",
            "total_chunks": total_chunks,
            "total_documents": len(docs),
            "documents": list(docs.values()),
        }, ensure_ascii=False, indent=2)

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "error": str(exc),
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

@_prompt()
def analisis_fundamental(
    emiten: str,
    tahun: int = 2023,
) -> str:
    """
    Generate a comprehensive fundamental analysis prompt for an IDX company.

    Produces a structured analysis template covering financial performance,
    valuation ratios, risk assessment, and investment outlook.

    Args:
        emiten: Company ticker code (e.g., BBCA, BBRI, BMRI).
        tahun:  Analysis year (default: 2023).
    """
    return f"""Lakukan analisis fundamental lengkap untuk {emiten.upper()} tahun {tahun}.

## Langkah Analisis

### 1. Kinerja Keuangan
Gunakan tool `tanya_laporan` dengan query:
- "Berapa laba bersih {emiten} {tahun}?"
- "Bagaimana pertumbuhan pendapatan {emiten} {tahun}?"
- "Berapa total aset dan ekuitas {emiten} {tahun}?"

### 2. Rasio Valuasi
Gunakan tool `hitung_rasio` dengan data dari laporan:
- PER (Price-to-Earnings Ratio)
- PBV (Price-to-Book Value)
- ROE (Return on Equity)
- DER (Debt-to-Equity Ratio)

### 3. Analisis Risiko
Gunakan tool `tanya_laporan` dengan query:
- "Apa saja faktor risiko utama {emiten} {tahun}?"
- "Bagaimana manajemen risiko kredit {emiten}?"

### 4. Prospek & Strategi
Gunakan tool `tanya_laporan` dengan query:
- "Apa target dan strategi {emiten} untuk tahun depan?"
- "Bagaimana outlook bisnis {emiten}?"

### 5. Ringkasan Eksekutif
Berikan ringkasan dalam format:
- **Kekuatan:** [3 poin utama]
- **Risiko:** [3 risiko utama]
- **Valuasi:** [Undervalued/Fairly Valued/Overvalued + alasan]
- **Rekomendasi:** [Beli/Tahan/Jual dengan disclaimer]

*Disclaimer: Analisis ini bukan rekomendasi investasi.*"""


@_prompt()
def deteksi_risiko(
    emiten: str,
    tahun: int = 2023,
    kategori_risiko: str = "semua",
) -> str:
    """
    Generate a risk detection analysis prompt for an IDX company.

    Creates a structured prompt for identifying and categorizing risks
    from annual reports: market risk, credit risk, operational risk,
    regulatory risk, and liquidity risk.

    Args:
        emiten:           Company ticker code.
        tahun:            Analysis year.
        kategori_risiko:  Risk category filter: market, credit, operational,
                          regulatory, liquidity, or 'semua' (all).
    """
    kategori_map = {
        "market":       "Risiko Pasar (suku bunga, nilai tukar, harga komoditas)",
        "credit":       "Risiko Kredit (NPL, kualitas aset, konsentrasi)",
        "operational":  "Risiko Operasional (sistem, fraud, SDM, siber)",
        "regulatory":   "Risiko Regulasi (perubahan kebijakan OJK/BI)",
        "liquidity":    "Risiko Likuiditas (LCR, NSFR, funding)",
        "semua":        "Semua Kategori Risiko",
    }

    kategori_label = kategori_map.get(
        kategori_risiko.lower(), kategori_map["semua"]
    )

    return f"""Lakukan deteksi dan pemetaan risiko untuk {emiten.upper()} tahun {tahun}.

## Fokus Analisis: {kategori_label}

### Langkah Deteksi Risiko

**Step 1: Ekstraksi dari Laporan**
Gunakan tool `tanya_laporan` dengan queries:
- "Faktor risiko utama yang dihadapi {emiten} {tahun}"
- "Risiko pasar dan dampaknya terhadap kinerja {emiten}"
- "Bagaimana {emiten} mengelola risiko kredit dan NPL?"
- "Risiko operasional dan siber yang diungkapkan {emiten}"
- "Rasio likuiditas LCR dan NSFR {emiten} {tahun}"

**Step 2: Ringkasan Laporan**
Gunakan tool `summarize_laporan` untuk mendapat gambaran risiko menyeluruh:
```
emiten: {emiten}
tahun: {tahun}
```

**Step 3: Pemetaan Risiko**
Buat matriks risiko dengan format:

| Kategori | Risiko Spesifik | Dampak (1-5) | Probabilitas (1-5) | Mitigasi |
|----------|----------------|--------------|-------------------|----------|
| Pasar    | ...            | ...          | ...               | ...      |
| Kredit   | ...            | ...          | ...               | ...      |
| Operasional | ...         | ...          | ...               | ...      |

**Step 4: Risk Score**
Hitung Risk Score = Dampak × Probabilitas untuk setiap risiko.
Identifikasi Top 3 risiko tertinggi yang perlu perhatian segera.

**Step 5: Kesimpulan**
- **Profil Risiko Keseluruhan:** [Rendah/Sedang/Tinggi]
- **Risiko Kritis:** [daftar risiko dengan score ≥ 15]
- **Rekomendasi Pemantauan:** [frekuensi dan metrik yang harus dipantau]"""


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    logger.info(
        "Starting FinSight IDX MCP Server | transport={}", transport
    )
    logger.info(
        "Tools: summarize_laporan, topic_modeling, tanya_laporan, hitung_rasio"
    )
    logger.info(
        "Resources: finsight://corpus"
    )
    logger.info(
        "Prompts: analisis_fundamental, deteksi_risiko"
    )

    mcp.run(transport=transport)
