"""
FinSight IDX — RAG Indexer
===========================
Chunks PDF documents, generates embeddings, and stores in ChromaDB.

Pipeline:
    PDF file
        → PDFExtractor (reuse from nlp/)
        → TextChunker  (sliding window with overlap)
        → SentenceTransformer embeddings
        → ChromaDB collection
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

from src.nlp.pdf_extractor import EXTRACTOR_VERSION, PAGE_BREAK, PDFExtractor
from src.rag.manifest import file_sha256, load_manifest, record_document


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION   = "finsight_documents"
DEFAULT_EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_SIZE   = 256    # characters per chunk (optimized for precision)
DEFAULT_CHUNK_OVERLAP = 64    # overlap between consecutive chunks


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single text chunk with metadata for RAG retrieval."""
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    section_title: str
    section_type: str
    chunk_index: int
    source_file: str
    company: str = ""
    year: int = 0
    char_count: int = 0
    extraction: str = "text"   # "text" or "ocr": OCR text can contain recognition errors
    chunk_type: str = "text"   # "text" (sliding window) or "table_row" (one table row)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(
                f"{self.document_id}_{self.chunk_index}".encode()
            ).hexdigest()[:16]

    def to_metadata(self) -> dict:
        """Convert to ChromaDB-compatible metadata (flat dict, no nested)."""
        return {
            "document_id":   self.document_id,
            "page_number":   self.page_number,
            "section_title": self.section_title[:100],
            "section_type":  self.section_type,
            "chunk_index":   self.chunk_index,
            "source_file":   self.source_file,
            "company":       self.company,
            "year":          self.year,
            "char_count":    self.char_count,
            "extraction":    self.extraction,
            "chunk_type":    self.chunk_type,
        }


@dataclass
class IndexStats:
    """Statistics from an indexing run."""
    document_id: str
    source_file: str
    total_pages: int
    total_chunks: int
    total_sections: int
    collection_name: str
    embed_model: str
    indexed_at: str = ""
    pages_ocr: list[int] = field(default_factory=list)
    pages_empty: list[int] = field(default_factory=list)
    boilerplate_lines_removed: int = 0


# ---------------------------------------------------------------------------
# Text Chunker
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Sliding-window chunker that respects sentence boundaries.

    Splits text into overlapping chunks for better retrieval recall.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks (text only)."""
        return [c for c, _ in self.chunk_with_offsets(text)]

    def chunk_with_offsets(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into overlapping chunks and return (chunk, start_offset).

        The cut is placed, in order of preference, at the end of a line, at the
        end of a sentence (". "), or at ``chunk_size``, and a line or sentence
        boundary is only accepted in the second half of the window. The next
        window starts ``overlap`` characters before the cut, moved forward to the
        start of a line when one is close, so table rows are not split and every
        character of ``text`` lands in at least one chunk.

        History:
        - v1 accepted a boundary right after ``start`` and advanced one
          character, producing runs of near-identical chunks (24,875 chunks,
          20,833 distinct, average 88 characters).
        - v2 (October 2026 audit) forced every step to advance
          ``chunk_size - overlap`` characters. When the cut fell in the first
          part of that distance the text in between was skipped: 1.8% of the
          BBCA 2023 report was never indexed. It also cut table rows mid-number.
        """
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [(text.strip().replace(PAGE_BREAK, "\n\n"), 0)]

        half = self.chunk_size // 2
        chunks: list[tuple[str, int]] = []
        start = 0
        n = len(text)
        while start < n:
            end = start + self.chunk_size
            if end >= n:
                piece = text[start:].strip()
                if piece:
                    chunks.append((piece.replace(PAGE_BREAK, "\n\n"), start))
                break
            line_end = max(text.rfind("\n", start + half, end), text.rfind(PAGE_BREAK, start + half, end))
            if line_end != -1:
                cut = line_end
            else:
                sentence = text.rfind(". ", start + half, end)
                cut = sentence + 1 if sentence != -1 else end
            piece = text[start:cut].strip()
            if piece:
                chunks.append((piece.replace(PAGE_BREAK, "\n\n"), start))
            # Next window: start at the earliest line start within `overlap` of the
            # cut (keeps rows whole); if the cut was itself a line end, the next
            # line; otherwise plain character overlap.
            lo = cut - self.overlap
            breaks = [i for i in (text.find("\n", lo, cut), text.find(PAGE_BREAK, lo, cut)) if i != -1]
            if breaks:
                nxt = min(breaks) + 1
            elif text[cut:cut + 1] in ("\n", PAGE_BREAK):
                nxt = cut + 1
            else:
                nxt = lo
            start = max(nxt, start + 1)
        return chunks


UNITS = re.compile(r"\((?:dalam|in)\s+(?:jutaan|miliar|ribuan|triliun|millions?|billions?)[^)]{0,40}\)", re.I)
YEAR = re.compile(r"(?<!\d)20\d\d(?!\d)")


def page_columns(page_text: str) -> tuple[str, list[str]]:
    """Unit and year columns of a table page, e.g. ("dalam miliar Rupiah", ["2023", "2022"])."""
    head = page_text.split("\n")[:10]          # a table's unit sits in its header
    units = UNITS.search("\n".join(head))
    years: list[str] = []
    for line in page_text.split("\n")[:15]:
        found = list(dict.fromkeys(YEAR.findall(line)))
        if len(found) >= 2 and len(line) < 120:   # a header row has two or more distinct years
            years = found
            break
    unit = ""
    if units:
        unit = re.sub(r"(jutaan|miliar|ribuan|triliun)(?=Rupiah)", r"\1 ", units.group(0).strip("()"))
        unit = re.sub(r"\s+", " ", unit)
    return unit, years


def page_context(page_text: str) -> str:
    """Short description of a table page: its unit and its year columns.

    A chunk cut from the middle of a table holds rows like
    "Total Aset 1.408.107 1.314.732 ..." with no unit and no year header.
    Prefixing the unit and the column years lets both the retriever and the
    model read the numbers correctly. Returns "" for pages without a table.
    """
    unit, years = page_columns(page_text)
    parts = [unit] if unit else []
    if years:
        parts.append("kolom " + " ".join(years))
    return " · ".join(parts)


VALUE = re.compile(r"^\(-?\d[\d.,]*\)%?$|^-?\d[\d.,]*%?$|^-$|^n\.a\.?$", re.I)
NUMBER = re.compile(r"\d[\d.,]*\d")
NOTE_REF = re.compile(r"^\d{1,2}[a-z]{0,2}(,\d{1,2}[a-z]{0,2})+,?$")   # "40,43", "2b,2g,4"
FOOTNOTE = re.compile(r"(?<=[A-Za-z)])\d{1,2}$")                         # "Total Kredit1", "(CAR)6"


def _clean_label(tokens: list[str]) -> str:
    while tokens and NOTE_REF.match(tokens[-1]):
        tokens = tokens[:-1]
    # A bare note number after a lower-case word ("laba rugi 43"), not "Tier 1".
    if len(tokens) >= 2 and re.fullmatch(r"\d{1,2}", tokens[-1]) and tokens[-2][:1].islower():
        tokens = tokens[:-1]
    return FOOTNOTE.sub("", " ".join(tokens)).strip(" :-,")


def table_rows(page_text: str, max_rows: int = 150) -> list[str]:
    """Turn the rows of a year-column table into self-contained sentences.

    "Total Aset 1.408.107 1.314.732 1.228.345" under the header
    "2023 2022 2021" becomes "Total Aset: 2023 = 1.408.107; 2022 = 1.314.732;
    2021 = 1.228.345 (dalam miliar Rupiah)". Rules, all chosen to skip a row
    rather than guess:
    - the number of trailing values must cover the year columns;
    - a label that itself contains several numbers (two tables printed side
      by side) is dropped;
    - a label that continues the line above ("(31 Desember 2022: Rp 743)")
      is replaced by the first line of that multi-line label, and values
      printed on their own line take the line above as their label.
    """
    unit, years = page_columns(page_text)
    if not years:
        return []
    rows: list[str] = []
    pending: list[str] = []          # label lines seen since the last row
    for line in page_text.split("\n"):
        tokens = line.split()
        n_vals = 0
        while n_vals < len(tokens) and VALUE.match(tokens[len(tokens) - 1 - n_vals]):
            n_vals += 1
        if n_vals < len(years):
            if line.strip() and len(set(YEAR.findall(line))) < 2:
                pending.append(line.strip())
            continue
        values = tokens[len(tokens) - len(years):]
        label = _clean_label(tokens[: len(tokens) - len(years)])
        if pending and not label:                       # values on their own line: label is the line above
            label = _clean_label(pending[-1].split())
        elif pending and (label[0] in "(" or label[0].islower() or label.lower().startswith("catatan")):
            label = _clean_label(pending[0 if len(pending) <= 3 else -1].split())   # wrapped label
        pending = []
        if (not label or YEAR.fullmatch(values[0])
                or len(NUMBER.findall(label)) >= 2 or len(label) > 140):
            continue
        cells = "; ".join(f"{y} = {v}" for y, v in zip(years, values))
        rows.append(f"{label}: {cells}" + (f" ({unit})" if unit else ""))
        if len(rows) >= max_rows:
            break
    return rows


def contextual_text(chunk_text: str, section_title: str, page: int, context: str) -> str:
    """Text that is embedded, searched by BM25 and shown to the model."""
    head = " · ".join(x for x in (section_title.strip()[:60], f"hal. {page}", context) if x)
    return f"[{head}]\n{chunk_text}"


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class RAGIndexer:
    """
    Indexes IDX annual report PDFs into ChromaDB for semantic retrieval.

    Usage:
        indexer = RAGIndexer(persist_dir="data/vectorstore")
        stats = indexer.index_pdf(
            "data/raw/bbca_laporan_tahunan_2023.pdf",
            company="BBCA",
            year=2023,
        )
        print(f"Indexed {stats.total_chunks} chunks")
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/vectorstore",
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        extractor: Optional[PDFExtractor] = None,
        index_table_rows: bool = True,
    ) -> None:
        if not HAS_CHROMA:
            raise ImportError("chromadb required. Run: pip install chromadb")
        if not HAS_ST:
            raise ImportError(
                "sentence-transformers required. "
                "Run: pip install sentence-transformers"
            )

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embed_model_name = embed_model
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.extractor = extractor or PDFExtractor()
        self.index_table_rows = index_table_rows

        self._embed_model: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_pdf(
        self,
        pdf_path: str | Path,
        company: str = "",
        year: int = 0,
        force_reindex: bool = False,
    ) -> IndexStats:
        """
        Extract, chunk, embed, and store a PDF in ChromaDB.

        Args:
            pdf_path:      Path to PDF file.
            company:       Company ticker/name (e.g., "BBCA").
            year:          Report year (e.g., 2023).
            force_reindex: If True, delete existing chunks and re-index.

        Returns:
            IndexStats with indexing summary.
        """
        path = Path(pdf_path)
        document_id = self._make_document_id(path, company, year)

        logger.info(
            "Starting indexing | file={} | company={} | year={}",
            path.name, company, year
        )

        # Check if already indexed
        collection = self._get_collection()
        # Look up existing chunks in both modes. v1 only did this when
        # force_reindex was False, so force_reindex=True raised NameError.
        existing = collection.get(
            where={"document_id": document_id},
            limit=1,
        )
        if not force_reindex:
            if existing["ids"]:
                # Fix: paginated count avoids SQL variables error on large collections
                count = self._count_document_chunks(collection, document_id)
                logger.info(
                    "Document already indexed | chunks={} | skipping",
                    count
                )
                entry = ((load_manifest(self.persist_dir) or {}).get("documents") or {}).get(document_id)
                if entry is None:
                    logger.warning("No manifest entry for this document: re-index with force_reindex=True")
                elif path.exists() and entry.get("pdf_sha256") != file_sha256(path):
                    logger.warning("PDF changed since it was indexed: re-index with force_reindex=True")
                from datetime import datetime
                return IndexStats(
                    document_id=document_id,
                    source_file=path.name,
                    total_pages=0,
                    total_chunks=count,
                    total_sections=0,
                    collection_name=self.collection_name,
                    embed_model=self.embed_model_name,
                    indexed_at=datetime.now().isoformat(),
                )

        # Extract PDF
        doc = self.extractor.extract(path)

        meta = getattr(doc, "metadata", None) or {}

        # Build chunks from sections
        all_chunks = self._build_chunks(doc, document_id, company, year, path.name)
        logger.info("Built {} chunks from {} sections", len(all_chunks), len(doc.sections))

        # Delete existing if force reindex
        if force_reindex and existing["ids"]:
            collection.delete(where={"document_id": document_id})

        # Embed and store in batches
        self._store_chunks(all_chunks)

        from datetime import datetime
        stats = IndexStats(
            document_id=document_id,
            source_file=path.name,
            total_pages=doc.total_pages,
            total_chunks=len(all_chunks),
            total_sections=len(doc.sections),
            collection_name=self.collection_name,
            embed_model=self.embed_model_name,
            indexed_at=datetime.now().isoformat(),
            pages_ocr=meta.get("pages_ocr", []),
            pages_empty=meta.get("pages_empty", []),
            boilerplate_lines_removed=meta.get("boilerplate_lines_removed", 0),
        )

        record_document(
            self.persist_dir,
            document_id,
            {
                "source_file": path.name,
                "company": company,
                "year": year,
                "pdf_sha256": file_sha256(path),
                "total_pages": doc.total_pages,
                "total_chunks": len(all_chunks),
                "extractor_version": meta.get("extractor_version", EXTRACTOR_VERSION),
                "ocr_used": meta.get("ocr_used", False),
                "pages_ocr": stats.pages_ocr,
                "pages_empty": stats.pages_empty,
                "boilerplate_lines_removed": stats.boilerplate_lines_removed,
                "table_row_chunks": sum(c.chunk_type == "table_row" for c in all_chunks),
            },
            embed_model=self.embed_model_name,
            chunk_size=self.chunker.chunk_size,
            chunk_overlap=self.chunker.overlap,
        )

        logger.success(
            "Indexing complete | chunks={} | pages={} | ocr_pages={} | empty_pages={}",
            stats.total_chunks, stats.total_pages, len(stats.pages_ocr), len(stats.pages_empty),
        )
        return stats

    def index_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Index raw text directly (without PDF extraction).
        Useful for testing or indexing pre-extracted content.

        Returns:
            Number of chunks indexed.
        """
        metadata = metadata or {}
        chunks_text = self.chunker.chunk(text)
        chunks = [
            DocumentChunk(
                chunk_id="",
                document_id=document_id,
                text=chunk,
                page_number=metadata.get("page_number", 0),
                section_title=metadata.get("section_title", ""),
                section_type=metadata.get("section_type", "general"),
                chunk_index=i,
                source_file=metadata.get("source_file", ""),
                company=metadata.get("company", ""),
                year=metadata.get("year", 0),
            )
            for i, chunk in enumerate(chunks_text)
        ]
        self._store_chunks(chunks)
        return len(chunks)

    def get_collection_stats(self) -> dict:
        """Return stats about the current ChromaDB collection."""
        collection = self._get_collection()
        count = collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_dir": str(self.persist_dir),
            "embed_model": self.embed_model_name,
        }

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns number deleted."""
        collection = self._get_collection()
        existing = collection.get(where={"document_id": document_id})
        if existing["ids"]:
            collection.delete(where={"document_id": document_id})
            logger.info(
                "Deleted {} chunks for document_id={}",
                len(existing["ids"]), document_id
            )
            return len(existing["ids"])
        return 0

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_collection(self):
        """Lazy-initialize ChromaDB client and collection."""
        if self._collection is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug(
                "ChromaDB collection ready | name={} | count={}",
                self.collection_name,
                self._collection.count(),
            )
        return self._collection

    def _get_embed_model(self) -> SentenceTransformer:
        """Lazy-load sentence transformer model."""
        if self._embed_model is None:
            logger.info("Loading embedding model: {}", self.embed_model_name)
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model

    def _build_chunks(
        self,
        doc,
        document_id: str,
        company: str,
        year: int,
        source_file: str,
    ) -> list[DocumentChunk]:
        """Build DocumentChunk list from ExtractedDocument sections."""
        all_chunks: list[DocumentChunk] = []
        global_index = 0
        page_source = {p.page_number: getattr(p, "source", "text") for p in getattr(doc, "pages", [])}
        page_ctx = {p.page_number: page_context(getattr(p, "text", "")) for p in getattr(doc, "pages", [])}

        for section in doc.sections:
            for chunk_text, offset in self.chunker.chunk_with_offsets(section.text):
                # Pages inside a section are joined with PAGE_BREAK, so the
                # number of breaks before the chunk gives its real page.
                page = section.start_page + section.text.count(PAGE_BREAK, 0, offset)
                chunk = DocumentChunk(
                    chunk_id="",
                    document_id=document_id,
                    text=contextual_text(chunk_text, section.title, page, page_ctx.get(page, "")),
                    page_number=page,
                    section_title=section.title,
                    section_type=section.section_type,
                    chunk_index=global_index,
                    source_file=source_file,
                    company=company,
                    year=year,
                    extraction="ocr" if page_source.get(page) == "ocr" else "text",
                )
                all_chunks.append(chunk)
                global_index += 1

        if getattr(self, "index_table_rows", True):
            # One extra chunk per table row, so a question about a single figure
            # can match a short text that holds exactly that label and value.
            title_of = {}
            for section in doc.sections:
                for pg in range(section.start_page, getattr(section, "end_page", section.start_page) + 1):
                    title_of.setdefault(pg, section.title)
            for p in getattr(doc, "pages", []):
                for row in table_rows(getattr(p, "text", "")):
                    all_chunks.append(DocumentChunk(
                        chunk_id="",
                        document_id=document_id,
                        text=contextual_text(row, title_of.get(p.page_number, ""), p.page_number, ""),
                        page_number=p.page_number,
                        section_title=title_of.get(p.page_number, ""),
                        section_type="financial",
                        chunk_index=global_index,
                        source_file=source_file,
                        company=company,
                        year=year,
                        extraction="ocr" if page_source.get(p.page_number) == "ocr" else "text",
                        chunk_type="table_row",
                    ))
                    global_index += 1

        return all_chunks

    def _store_chunks(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 100,
    ) -> None:
        """Embed and store chunks in ChromaDB in batches."""
        if not chunks:
            return

        collection = self._get_collection()
        embed_model = self._get_embed_model()

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]

            logger.debug(
                "Embedding batch {}/{} | size={}",
                i // batch_size + 1,
                (len(chunks) - 1) // batch_size + 1,
                len(batch),
            )

            embeddings = embed_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[c.to_metadata() for c in batch],
            )

        logger.debug("Stored {} chunks in ChromaDB", len(chunks))

    @staticmethod
    def _make_document_id(path: Path, company: str, year: int) -> str:
        """Generate stable document ID from file path + metadata."""
        key = f"{path.stem}_{company}_{year}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    @staticmethod
    def _count_document_chunks(collection, document_id: str) -> int:
        """
        Count chunks for a document using pagination to avoid
        'too many SQL variables' error on large ChromaDB collections.
        """
        count = 0
        offset = 0
        page_size = 500
        while True:
            batch = collection.get(
                where={"document_id": document_id},
                limit=page_size,
                offset=offset,
                include=[],
            )
            count += len(batch["ids"])
            if len(batch["ids"]) < page_size:
                break
            offset += page_size
        return count
