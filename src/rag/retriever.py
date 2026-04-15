"""
FinSight IDX — RAG Retriever
==============================
Hybrid retrieval combining BM25 (lexical) and semantic (vector) search.

Strategy:
    1. Semantic search   — ChromaDB cosine similarity (dense retrieval)
    2. BM25 search       — keyword matching (sparse retrieval)
    3. Reciprocal Rank Fusion (RRF) — merge and rerank results
    4. Return top-k chunks with unified relevance scores
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

from src.rag.indexer import RAGIndexer, DEFAULT_EMBED_MODEL, DEFAULT_COLLECTION


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance metadata."""
    chunk_id: str
    text: str
    page_number: int
    section_title: str
    section_type: str
    source_file: str
    company: str
    year: int
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0

    def citation(self) -> str:
        """Generate a citation string for this chunk."""
        parts = []
        if self.company:
            parts.append(self.company)
        if self.year:
            parts.append(str(self.year))
        if self.section_title:
            parts.append(self.section_title[:40])
        parts.append(f"hal. {self.page_number}")
        return f"[{', '.join(parts)}]"


# ---------------------------------------------------------------------------
# BM25 Implementation (pure Python, no extra dependency)
# ---------------------------------------------------------------------------

class BM25:
    """
    Lightweight BM25 scorer for a fixed corpus of documents.
    Used for lexical retrieval over retrieved semantic candidates.

    Parameters follow Robertson et al. (k1=1.5, b=0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def fit(self, documents: list[str]) -> None:
        """Build BM25 index from a list of documents."""
        self._corpus = [self._tokenize(d) for d in documents]
        self._doc_freqs = [
            self._term_freq(tokens) for tokens in self._corpus
        ]
        self._avgdl = (
            sum(len(t) for t in self._corpus) / len(self._corpus)
            if self._corpus else 1.0
        )
        self._idf = self._compute_idf()

    def score(self, query: str, doc_index: int) -> float:
        """Compute BM25 score for a query against one document."""
        query_tokens = self._tokenize(query)
        doc_tokens = self._corpus[doc_index]
        doc_len = len(doc_tokens)
        tf = self._doc_freqs[doc_index]
        score = 0.0
        for token in query_tokens:
            if token not in tf:
                continue
            idf = self._idf.get(token, 0.0)
            freq = tf[token]
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_len / self._avgdl
            )
            score += idf * (numerator / denominator)
        return score

    def score_all(self, query: str) -> list[float]:
        """Compute BM25 scores for all documents."""
        return [self.score(query, i) for i in range(len(self._corpus))]

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer."""
        return text.lower().split()

    def _term_freq(self, tokens: list[str]) -> dict[str, int]:
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        return dict(tf)

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._corpus)
        df: dict[str, int] = defaultdict(int)
        for doc_freq in self._doc_freqs:
            for term in doc_freq:
                df[term] += 1
        return {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Hybrid retriever combining semantic (ChromaDB) and BM25 search.

    Usage:
        retriever = RAGRetriever(persist_dir="data/vectorstore")
        results = retriever.retrieve("Berapa laba bersih BCA 2023?", top_k=5)
        for chunk in results:
            print(chunk.citation(), chunk.text[:100])
    """

    def __init__(
        self,
        persist_dir: str = "data/vectorstore",
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        semantic_candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        """
        Args:
            persist_dir:         ChromaDB storage directory.
            collection_name:     ChromaDB collection name.
            embed_model:         SentenceTransformer model name.
            semantic_candidates: Number of semantic results before RRF rerank.
            rrf_k:               RRF constant (higher = smoother ranking).
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model
        self.semantic_candidates = semantic_candidates
        self.rrf_k = rrf_k

        self._indexer = RAGIndexer(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embed_model=embed_model,
        )
        self._embed_model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        section_type_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid retrieval: semantic + BM25 with RRF reranking.

        Args:
            query:               Natural language query.
            top_k:               Number of results to return.
            company_filter:      Filter by company ticker (e.g., "BBCA").
            year_filter:         Filter by report year (e.g., 2023).
            section_type_filter: Filter by section type ("financial", "risk", etc.)

        Returns:
            List of RetrievedChunk sorted by RRF score (highest first).

        Raises:
            ValueError: If the collection is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        collection = self._indexer._get_collection()
        if collection.count() == 0:
            raise ValueError(
                "ChromaDB collection is empty. "
                "Run RAGIndexer.index_pdf() first."
            )

        # Build filter
        where_filter = self._build_filter(
            company_filter, year_filter, section_type_filter
        )

        # Step 1: Semantic search
        semantic_results = self._semantic_search(
            query, self.semantic_candidates, where_filter
        )
        logger.debug(
            "Semantic search returned {} candidates", len(semantic_results)
        )

        if not semantic_results:
            return []

        # Step 2: BM25 over semantic candidates
        bm25_results = self._bm25_search(query, semantic_results)

        # Step 3: RRF fusion
        fused = self._reciprocal_rank_fusion(semantic_results, bm25_results)

        # Return top_k
        results = fused[:top_k]
        for i, chunk in enumerate(results):
            chunk.rank = i + 1

        logger.info(
            "Retrieved {} chunks for query: '{}'",
            len(results), query[:60]
        )
        return results

    def retrieve_semantic_only(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[dict] = None,
    ) -> list[RetrievedChunk]:
        """Semantic-only retrieval (faster, no BM25)."""
        return self._semantic_search(query, top_k, where_filter)

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _semantic_search(
        self,
        query: str,
        n_results: int,
        where_filter: Optional[dict],
    ) -> list[RetrievedChunk]:
        """Query ChromaDB with dense vector similarity."""
        embed_model = self._get_embed_model()
        query_embedding = embed_model.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        collection = self._indexer._get_collection()

        kwargs: dict = {
            "query_embeddings": query_embedding,
            "n_results": min(n_results, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = collection.query(**kwargs)

        chunks: list[RetrievedChunk] = []
        for i, (doc_id, text, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB cosine distance → similarity score
            semantic_score = 1.0 - dist

            chunks.append(RetrievedChunk(
                chunk_id=doc_id,
                text=text,
                page_number=meta.get("page_number", 0),
                section_title=meta.get("section_title", ""),
                section_type=meta.get("section_type", ""),
                source_file=meta.get("source_file", ""),
                company=meta.get("company", ""),
                year=meta.get("year", 0),
                semantic_score=round(semantic_score, 4),
            ))
        return chunks

    def _bm25_search(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Re-rank candidates using BM25 lexical scoring."""
        if not candidates:
            return []

        bm25 = BM25()
        bm25.fit([c.text for c in candidates])
        scores = bm25.score_all(query)

        # Pair chunks with BM25 scores and sort
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for chunk, score in scored:
            chunk.bm25_score = round(score, 4)
            result.append(chunk)
        return result

    def _reciprocal_rank_fusion(
        self,
        semantic_list: list[RetrievedChunk],
        bm25_list: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        RRF(d) = Σ 1 / (k + rank(d))
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        for rank, chunk in enumerate(semantic_list, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        for rank, chunk in enumerate(bm25_list, start=1):
            rrf_scores[chunk.chunk_id] += 1.0 / (self.rrf_k + rank)

        # Build lookup
        chunk_lookup = {c.chunk_id: c for c in semantic_list}

        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        result = []
        for cid in sorted_ids:
            if cid in chunk_lookup:
                chunk = chunk_lookup[cid]
                chunk.rrf_score = round(rrf_scores[cid], 6)
                result.append(chunk)

        return result

    def _build_filter(
        self,
        company: Optional[str],
        year: Optional[int],
        section_type: Optional[str],
    ) -> Optional[dict]:
        """Build ChromaDB where filter from optional parameters."""
        conditions = []
        if company:
            conditions.append({"company": {"$eq": company}})
        if year:
            conditions.append({"year": {"$eq": year}})
        if section_type:
            conditions.append({"section_type": {"$eq": section_type}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _get_embed_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embed_model is None:
            logger.info("Loading embedding model: {}", self.embed_model_name)
            self._embed_model = SentenceTransformer(self.embed_model_name)
        return self._embed_model
