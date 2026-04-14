"""
FinSight IDX — Financial Topic Modeler
========================================
Neural topic modeling for Indonesian financial news using BERTopic
with multilingual sentence-transformers embeddings.

Pipeline:
    List[str] (article texts)
        → SentenceTransformer embeddings
        → UMAP dimensionality reduction
        → HDBSCAN clustering
        → BERTopic topic extraction
        → TopicModelResult (topics, distributions, drift)
        → JSON / CSV output
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# BERTopic stack — imported lazily to allow unit testing without full install
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    HAS_BERTOPIC = True
except ImportError as _e:
    HAS_BERTOPIC = False
    _IMPORT_ERROR = str(_e)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Best multilingual model for Indonesian — supports 50+ languages including ID
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Financial domain stopwords (Indonesian + English)
FINANCIAL_STOPWORDS = [
    # Indonesian function words
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "akan", "telah", "dalam", "tidak", "juga", "atau",
    "sebagai", "oleh", "tersebut", "dapat", "lebih", "tahun", "bulan",
    "per", "triwulan", "kuartal", "tbk", "pt", "rp",
    # English function words
    "the", "a", "an", "of", "in", "to", "and", "is", "for", "that",
]


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TopicInfo:
    """Information about a single discovered topic."""
    topic_id: int
    label: str
    size: int                          # number of documents
    representative_words: list[str]
    representative_docs: list[str]     # sample document texts
    probability: float = 0.0


@dataclass
class TopicModelResult:
    """Full output of topic modeling run."""
    model_name: str
    processed_at: str
    n_documents: int
    n_topics: int
    embedding_model: str
    topics: list[TopicInfo]
    topic_assignments: list[int]       # topic id per document
    outlier_count: int = 0             # docs assigned to topic -1
    topic_over_time: Optional[list[dict]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["topics"] = [asdict(t) for t in self.topics]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Topic Model: {self.model_name}",
            f"Documents  : {self.n_documents}",
            f"Topics     : {self.n_topics} (+ outliers: {self.outlier_count})",
            "",
            "Top Topics:",
        ]
        for t in sorted(self.topics, key=lambda x: x.size, reverse=True)[:10]:
            words = ", ".join(t.representative_words[:5])
            lines.append(f"  [{t.topic_id:3d}] {t.label:30s} | {t.size:4d} docs | {words}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Topic Modeler
# ---------------------------------------------------------------------------

class FinancialTopicModeler:
    """
    Neural topic modeling for Indonesian financial news.

    Usage:
        modeler = FinancialTopicModeler()
        texts = ["BCA laba naik 20%...", "BI naikkan suku bunga...", ...]
        result = modeler.fit(texts)
        print(result.summary())
        modeler.save(result, "data/processed/topics_2023.json")
    """

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        n_topics: str | int = "auto",
        min_topic_size: int = 5,
        output_dir: str | Path = "data/processed",
    ) -> None:
        """
        Args:
            embedding_model: SentenceTransformer model name.
            n_topics:        Number of topics or 'auto' for automatic detection.
            min_topic_size:  Minimum documents per topic (HDBSCAN parameter).
            output_dir:      Directory for saving results.
        """
        if not HAS_BERTOPIC:
            raise ImportError(
                f"BERTopic stack not available: {_IMPORT_ERROR}\n"
                "Run: pip install bertopic sentence-transformers umap-learn hdbscan"
            )

        self.embedding_model_name = embedding_model
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_model: Optional[SentenceTransformer] = None
        self._topic_model: Optional[BERTopic] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        texts: list[str],
        timestamps: Optional[list[str]] = None,
        model_name: str = "finsight_topics",
    ) -> TopicModelResult:
        """
        Fit topic model on a list of text documents.

        Args:
            texts:      List of article/document texts. Min 50 recommended.
            timestamps: ISO date strings per document (for topic drift).
                        Must be same length as texts if provided.
            model_name: Label for the output result.

        Returns:
            TopicModelResult with discovered topics and assignments.

        Raises:
            ValueError: If fewer than 10 documents provided.
            ImportError: If BERTopic dependencies not installed.
        """
        if len(texts) < 10:
            raise ValueError(
                f"Need at least 10 documents, got {len(texts)}."
            )

        logger.info(
            "Starting topic modeling | docs={} | model={}",
            len(texts), self.embedding_model_name
        )

        # Step 1: Load embedding model
        embedding_model = self._get_embedding_model()

        # Step 2: Build BERTopic with custom components
        topic_model = self._build_topic_model()

        # Step 3: Fit
        logger.info("Fitting BERTopic model...")
        topic_assignments, probabilities = topic_model.fit_transform(
            texts,
            embeddings=None,  # let BERTopic compute internally
        )

        self._topic_model = topic_model
        logger.success("BERTopic fitting complete.")

        # Step 4: Extract topic info
        topics = self._extract_topic_info(topic_model, texts, topic_assignments)

        # Step 5: Optional topic over time
        topic_over_time = None
        if timestamps:
            topic_over_time = self._compute_topic_drift(
                topic_model, texts, topic_assignments, timestamps
            )

        # Step 6: Build result
        outlier_count = sum(1 for t in topic_assignments if t == -1)
        result = TopicModelResult(
            model_name=model_name,
            processed_at=datetime.now().isoformat(),
            n_documents=len(texts),
            n_topics=len(topics),
            embedding_model=self.embedding_model_name,
            topics=topics,
            topic_assignments=list(topic_assignments),
            outlier_count=outlier_count,
            topic_over_time=topic_over_time,
        )

        logger.success(
            "Topic modeling complete | topics={} | outliers={}",
            len(topics), outlier_count
        )
        return result

    def save(
        self,
        result: TopicModelResult,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save TopicModelResult as JSON.

        Args:
            result:   Result to save.
            filename: Output filename. Auto-generated if None.

        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"topics_{timestamp}.json"

        output_path = self.output_dir / filename
        output_path.write_text(result.to_json(), encoding="utf-8")
        logger.success("Topic model result saved → {}", output_path)
        return output_path

    def save_csv(
        self,
        result: TopicModelResult,
        texts: list[str],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save document-topic assignments as CSV for analysis.

        Args:
            result: TopicModelResult from fit().
            texts:  Original document texts (same order as fit()).
            filename: Output filename.

        Returns:
            Path to saved CSV file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"topic_assignments_{timestamp}.csv"

        topic_labels = {t.topic_id: t.label for t in result.topics}
        topic_labels[-1] = "Outlier"

        rows = []
        for i, (text, topic_id) in enumerate(
            zip(texts, result.topic_assignments)
        ):
            rows.append({
                "doc_id": i,
                "topic_id": topic_id,
                "topic_label": topic_labels.get(topic_id, "Unknown"),
                "text_preview": text[:150],
            })

        df = pd.DataFrame(rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.success("Topic assignments CSV saved → {}", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _get_embedding_model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model."""
        if self._embedding_model is None:
            logger.info(
                "Loading embedding model: {}", self.embedding_model_name
            )
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )
        return self._embedding_model

    def _build_topic_model(self) -> BERTopic:
        """Construct BERTopic with tuned components for financial text."""
        umap_model = UMAP(
            n_neighbors=10,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            stop_words=FINANCIAL_STOPWORDS,
            min_df=2,
            ngram_range=(1, 2),  # unigrams + bigrams
        )

        n_topics = None if self.n_topics == "auto" else self.n_topics

        return BERTopic(
            embedding_model=self._get_embedding_model(),
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=n_topics,
            top_n_words=10,
            verbose=False,
        )

    def _extract_topic_info(
        self,
        topic_model: BERTopic,
        texts: list[str],
        assignments: list[int],
    ) -> list[TopicInfo]:
        """Extract structured TopicInfo from fitted BERTopic model."""
        topic_info_df = topic_model.get_topic_info()
        topics: list[TopicInfo] = []

        for _, row in topic_info_df.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue  # Skip outlier cluster

            # Get representative words
            words_scores = topic_model.get_topic(tid) or []
            words = [w for w, _ in words_scores[:8]]

            # Get representative documents
            rep_docs = topic_model.get_representative_docs(tid) or []
            rep_docs_preview = [d[:200] for d in rep_docs[:3]]

            # Auto-label based on top words
            label = self._auto_label(words)

            # Calculate topic probability (proportion of docs)
            topic_doc_count = sum(1 for a in assignments if a == tid)
            prob = topic_doc_count / len(assignments) if assignments else 0.0

            topics.append(TopicInfo(
                topic_id=tid,
                label=label,
                size=int(row.get("Count", topic_doc_count)),
                representative_words=words,
                representative_docs=rep_docs_preview,
                probability=round(prob, 4),
            ))

        return sorted(topics, key=lambda t: t.size, reverse=True)

    def _compute_topic_drift(
        self,
        topic_model: BERTopic,
        texts: list[str],
        assignments: list[int],
        timestamps: list[str],
    ) -> list[dict]:
        """Compute topic frequency over time for trend analysis."""
        try:
            topics_over_time = topic_model.topics_over_time(
                texts,
                timestamps,
                nr_bins=10,
                evolution_tuning=True,
            )
            return topics_over_time.to_dict(orient="records")
        except Exception as exc:
            logger.warning("Topic drift computation failed: {}", exc)
            return []

    @staticmethod
    def _auto_label(words: list[str]) -> str:
        """
        Generate a human-readable topic label from representative words.
        Uses keyword matching for common Indonesian financial topics.
        """
        word_set = set(w.lower() for w in words)

        LABEL_RULES: list[tuple[set, str]] = [
            ({"suku", "bunga", "bi", "rate", "acuan"}, "Suku Bunga BI"),
            ({"inflasi", "harga", "cpi", "konsumen"}, "Inflasi & Harga"),
            ({"ihsg", "saham", "bursa", "indeks", "trading"}, "Pasar Saham IHSG"),
            ({"kredit", "pinjaman", "loan", "npf", "npl"}, "Kredit Perbankan"),
            ({"rupiah", "kurs", "valas", "usd", "dolar"}, "Nilai Tukar Rupiah"),
            ({"laba", "pendapatan", "revenue", "profit", "kinerja"}, "Kinerja Keuangan"),
            ({"batubara", "nikel", "cpo", "sawit", "komoditas"}, "Komoditas"),
            ({"obligasi", "sbn", "yield", "surat", "negara"}, "Obligasi & SBN"),
            ({"aset", "merger", "akuisisi", "ekspansi"}, "Aksi Korporasi"),
            ({"ojk", "regulasi", "kebijakan", "otoritas"}, "Regulasi Keuangan"),
            ({"digital", "fintech", "teknologi", "aplikasi"}, "Digital & Fintech"),
            ({"umkm", "usaha", "mikro", "kecil"}, "UMKM & Wirausaha"),
        ]

        for keywords, label in LABEL_RULES:
            if word_set & keywords:
                return label

        # Fallback: use top 3 words as label
        return " | ".join(words[:3]).title()
