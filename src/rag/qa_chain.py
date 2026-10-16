"""
FinSight IDX — RAG QA Chain
=============================
Retrieval-Augmented Generation for financial document Q&A.

Pipeline:
    User question
        → RAGRetriever.retrieve()   (hybrid search)
        → build_context()           (format chunks with citations)
        → Claude API                (generate answer with citations)
        → QAResponse                (answer + sources + metadata)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.api.client import get_client, sampling_kwargs
from src.rag.retriever import RAGRetriever, RetrievedChunk
from src.config import claude_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """Anda adalah analis keuangan ahli yang menjawab pertanyaan
tentang laporan keuangan perusahaan-perusahaan yang terdaftar di Bursa Efek Indonesia (IDX).

Aturan penting:
1. Jawab HANYA berdasarkan konteks yang diberikan — jangan mengarang fakta
2. Setiap klaim WAJIB disertai citation dalam format [Sumber: nama, hal. X]
3. Jika informasi tidak ada dalam konteks, jawab "Informasi tidak tersedia dalam dokumen"
4. Gunakan angka yang tepat dari dokumen, bukan estimasi, dan selalu sebutkan satuannya
   (juta/miliar/triliun Rupiah atau persen) sesuai dokumen
5. Baris tabel di konteks berbentuk "Label: 2023 = x; 2022 = y (satuan)": ambil kolom
   tahun yang ditanyakan
6. Utamakan angka konsolidasian (Ikhtisar Data Keuangan, laporan keuangan konsolidasian)
   dibanding angka entitas induk saja, anak perusahaan, atau satu segmen, kecuali ditanya
7. Jika konteks memuat beberapa angka berbeda untuk hal yang sama, sebutkan semuanya
   beserta halamannya, jangan memilih salah satu diam-diam
8. Jawab dalam Bahasa Indonesia yang formal dan ringkas"""

QA_USER_TEMPLATE = """Konteks dari laporan keuangan IDX:
{context}

---
Pertanyaan: {question}

Jawab pertanyaan di atas berdasarkan konteks. Sertakan citation [Sumber: ...] 
untuk setiap fakta penting yang kamu sebutkan."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class QASource:
    """A source document used in generating the answer."""
    chunk_id: str
    citation: str
    text_preview: str
    page_number: int
    section_title: str
    company: str
    year: int
    relevance_score: float


@dataclass
class QAResponse:
    """Full response from the RAG QA chain."""
    question: str
    answer: str
    sources: list[QASource]
    model: str
    retrieved_chunks: int
    answered_at: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sources"] = [asdict(s) for s in self.sources]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def format_answer(self) -> str:
        """Human-readable formatted answer with sources."""
        lines = [
            f"❓ {self.question}",
            "",
            f"💡 {self.answer}",
            "",
            "📚 Sumber:",
        ]
        for i, src in enumerate(self.sources, 1):
            lines.append(
                f"  [{i}] {src.citation} "
                f"— \"{src.text_preview[:100]}...\""
            )
        lines.append(
            f"\n⚡ {self.input_tokens} in / {self.output_tokens} out tokens "
            f"| {self.latency_ms:.0f}ms"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QA Chain
# ---------------------------------------------------------------------------

class FinancialQAChain:
    """
    RAG-powered Q&A chain for IDX financial documents.

    Usage:
        # First index a document
        from src.rag.indexer import RAGIndexer
        indexer = RAGIndexer()
        indexer.index_pdf("data/raw/bbca_laporan_tahunan_2023.pdf",
                          company="BBCA", year=2023)

        # Then query
        qa = FinancialQAChain()
        response = qa.ask("Berapa laba bersih BCA tahun 2023?")
        print(response.format_answer())
    """

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        model: str | None = None,
        top_k: int | None = None,
        max_context_chars: int = 6000,
        retrieval_mode: str | None = None,
    ) -> None:
        """
        Args:
            persist_dir:       ChromaDB storage directory.
            model:             Claude model for answer generation.
            top_k:             Number of chunks to retrieve.
            max_context_chars: Max characters of context sent to Claude.
        """
        self.model = model or claude_model()
        self.top_k = top_k or int(os.getenv("RAG_TOP_K", "5"))
        self.retrieval_mode = (retrieval_mode or os.getenv("RETRIEVAL_MODE", "hybrid")).lower()
        if self.retrieval_mode not in {"hybrid", "bm25"}:
            raise ValueError("retrieval_mode must be 'hybrid' or 'bm25'")
        self.max_context_chars = max_context_chars
        self.retriever = RAGRetriever(persist_dir=persist_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        section_type_filter: Optional[str] = None,
    ) -> QAResponse:
        """
        Answer a financial question using RAG.

        Args:
            question:            Natural language question.
            company_filter:      Limit search to specific company (e.g., "BBCA").
            year_filter:         Limit search to specific year (e.g., 2023).
            section_type_filter: Limit to section type ("financial", "risk", etc.)

        Returns:
            QAResponse with answer, citations, and token usage.

        Raises:
            ValueError: If question is empty or collection is empty.
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        start_time = datetime.now()
        logger.info("Processing question: '{}'", question[:80])

        # Step 1: Retrieve relevant chunks
        if getattr(self, "retrieval_mode", "hybrid") == "bm25":
            where = self.retriever._build_filter(company_filter, year_filter, section_type_filter)
            chunks = self.retriever._bm25_search_corpus(question, where, self.top_k)
        else:
            chunks = self.retriever.retrieve(
                query=question,
                top_k=self.top_k,
                company_filter=company_filter,
                year_filter=year_filter,
                section_type_filter=section_type_filter,
            )

        if not chunks:
            return self._no_results_response(question)

        # Step 2: Build context with numbered citations
        context, citation_map = self._build_context(chunks)

        # Step 3: Call Claude
        response = get_client().messages.create(
            model=self.model,
            max_tokens=1024,
            **sampling_kwargs(0.1),
            system=QA_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": QA_USER_TEMPLATE.format(
                    context=context,
                    question=question,
                ),
            }],
        )

        answer = response.content[0].text
        latency = (datetime.now() - start_time).total_seconds() * 1000

        # Step 4: Build sources list
        sources = [
            QASource(
                chunk_id=chunk.chunk_id,
                citation=chunk.citation(),
                text_preview=chunk.text[:150],
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                company=chunk.company,
                year=chunk.year,
                relevance_score=round(chunk.rrf_score, 4),
            )
            for chunk in chunks
        ]

        logger.success(
            "Answer generated | tokens={}/{} | latency={:.0f}ms",
            response.usage.input_tokens,
            response.usage.output_tokens,
            latency,
        )

        return QAResponse(
            question=question,
            answer=answer,
            sources=sources,
            model=self.model,
            retrieved_chunks=len(chunks),
            answered_at=datetime.now().isoformat(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=round(latency, 1),
        )

    def ask_batch(
        self,
        questions: list[str],
        **kwargs,
    ) -> list[QAResponse]:
        """
        Answer multiple questions sequentially.

        Args:
            questions: List of question strings.
            **kwargs:  Passed to ask().

        Returns:
            List of QAResponse objects.
        """
        responses = []
        for i, q in enumerate(questions):
            logger.info("Batch Q&A {}/{}", i + 1, len(questions))
            response = self.ask(q, **kwargs)
            responses.append(response)
        return responses

    def save_response(
        self,
        response: QAResponse,
        output_dir: str | Path = "data/processed",
    ) -> Path:
        """Save QAResponse as JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_response_{ts}.json"
        path = output_path / filename
        path.write_text(response.to_json(), encoding="utf-8")
        logger.info("Response saved → {}", path)
        return path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, dict[int, RetrievedChunk]]:
        """
        Format chunks into numbered context string for Claude.

        Returns:
            Tuple of (context_string, citation_map {number: chunk})
        """
        context_parts: list[str] = []
        citation_map: dict[int, RetrievedChunk] = {}
        total_chars = 0

        for i, chunk in enumerate(chunks, start=1):
            citation_map[i] = chunk
            part = (
                f"[{i}] {chunk.citation()}\n"
                f"{chunk.text}"
            )

            if total_chars + len(part) > self.max_context_chars:
                logger.debug(
                    "Context limit reached at chunk {}/{}", i, len(chunks)
                )
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n\n---\n\n".join(context_parts), citation_map

    def _no_results_response(self, question: str) -> QAResponse:
        """Return a response when no chunks are retrieved."""
        return QAResponse(
            question=question,
            answer=(
                "Maaf, tidak ditemukan informasi yang relevan dalam dokumen "
                "yang terindeks. Pastikan dokumen sudah diindeks terlebih dahulu."
            ),
            sources=[],
            model=self.model,
            retrieved_chunks=0,
            answered_at=datetime.now().isoformat(),
        )
