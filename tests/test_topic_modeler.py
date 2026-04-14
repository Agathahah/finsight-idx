"""
Tests for src/nlp/topic_modeler.py and src/nlp/news_scraper.py
===============================================================
Uses mocks for BERTopic — no real model training needed.
Tests news_scraper dataset generator without network calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

from src.nlp.news_scraper import (  # noqa: E402
    NewsArticle,
    FinancialNewsScraper,
    generate_sample_dataset,
)
from src.nlp.topic_modeler import (  # noqa: E402
    TopicInfo,
    TopicModelResult,
    FinancialTopicModeler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "Bank Indonesia mempertahankan suku bunga acuan di level 6% untuk menjaga stabilitas.",
    "IHSG menguat 1,2% didorong sektor perbankan dan consumer goods pada perdagangan hari ini.",
    "Inflasi Maret tercatat 3,05% yoy, naik tipis dari bulan sebelumnya sebesar 2,98%.",
    "BCA membukukan laba bersih Rp 48,6 triliun tumbuh 19,5% pada tahun 2023.",
    "Rupiah melemah ke level Rp 15.800 per USD di tengah penguatan indeks dolar global.",
    "OJK mencatat kredit perbankan tumbuh 10,4% yoy didorong segmen korporasi dan UMKM.",
    "Harga batubara internasional turun 5% mempengaruhi kinerja emiten tambang di BEI.",
    "Pemerintah lelang SBN senilai Rp 25 triliun dengan yield 6,8% untuk seri benchmark.",
    "BBRI mencatat pertumbuhan kredit mikro 15% menjadi Rp 350 triliun per Desember 2023.",
    "BI catat cadangan devisa USD 144 miliar setara 7,5 bulan impor dan pembayaran utang.",
    "Fintech P2P lending salurkan pinjaman Rp 60 triliun, OJK perketat aturan bunga.",
    "Merger Bank Syariah Indonesia perkuat posisi perbankan syariah nasional dengan aset Rp 350 T.",
]

SAMPLE_TIMESTAMPS = [
    "2023-01-15", "2023-02-10", "2023-03-05", "2023-04-20",
    "2023-05-15", "2023-06-08", "2023-07-22", "2023-08-14",
    "2023-09-03", "2023-10-18", "2023-11-25", "2023-12-10",
]


def _mock_bertopic(texts: list[str]) -> tuple:
    """Return mock topic assignments and probabilities."""
    n = len(texts)
    assignments = [i % 4 for i in range(n)]  # 4 topics
    probs = [[0.7, 0.1, 0.1, 0.1]] * n
    return assignments, probs


def _mock_topic_info_df():
    import pandas as pd
    return pd.DataFrame({
        "Topic": [0, 1, 2, 3],
        "Count": [4, 3, 3, 2],
        "Name": ["0_suku_bunga", "1_ihsg", "2_inflasi", "3_kredit"],
    })


# ---------------------------------------------------------------------------
# NewsArticle Tests
# ---------------------------------------------------------------------------

class TestNewsArticle:

    def test_auto_generates_article_id(self):
        article = NewsArticle(
            article_id="",
            title="Test",
            text="Some financial news text here.",
            source="kontan",
            url="https://kontan.co.id/article/123",
            published_at="2023-01-01T00:00:00+00:00",
        )
        assert article.article_id != ""
        assert len(article.article_id) == 12

    def test_char_count_computed(self):
        text = "Berita keuangan Indonesia hari ini sangat menarik."
        article = NewsArticle(
            article_id="abc123",
            title="Test",
            text=text,
            source="test",
            url="https://example.com",
            published_at="2023-01-01",
        )
        assert article.char_count == len(text)

    def test_keeps_provided_article_id(self):
        article = NewsArticle(
            article_id="myid_001",
            title="T",
            text="text",
            source="s",
            url="https://x.com",
            published_at="2023-01-01",
        )
        assert article.article_id == "myid_001"


# ---------------------------------------------------------------------------
# generate_sample_dataset Tests
# ---------------------------------------------------------------------------

class TestGenerateSampleDataset:

    def test_generates_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_sample_dataset(n_articles=50, output_dir=tmpdir)
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 50

    def test_output_is_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_sample_dataset(n_articles=10, output_dir=tmpdir)
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                obj = json.loads(line)
                assert "title" in obj
                assert "text" in obj
                assert "source" in obj
                assert obj["source"] == "synthetic"

    def test_generates_500_articles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_sample_dataset(n_articles=500, output_dir=tmpdir)
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 500

    def test_articles_have_indonesian_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_sample_dataset(n_articles=20, output_dir=tmpdir)
            texts = [
                json.loads(l)["text"]
                for l in path.read_text().strip().split("\n")
            ]
            # At least some articles should contain Indonesian financial terms
            indonesian_terms = ["suku bunga", "inflasi", "rupiah", "kredit", "IHSG"]
            found = sum(
                1 for text in texts
                if any(term.lower() in text.lower() for term in indonesian_terms)
            )
            assert found > 0

    def test_published_dates_in_2023(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_sample_dataset(n_articles=20, output_dir=tmpdir)
            for line in path.read_text().strip().split("\n"):
                obj = json.loads(line)
                assert obj["published_at"].startswith("2023")


# ---------------------------------------------------------------------------
# FinancialNewsScraper Tests
# ---------------------------------------------------------------------------

class TestFinancialNewsScraper:

    def test_save_jsonl(self):
        articles = [
            NewsArticle(
                article_id=f"id_{i}",
                title=f"Title {i}",
                text=f"Financial news text number {i}",
                source="test",
                url=f"https://example.com/{i}",
                published_at="2023-01-01",
            )
            for i in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = FinancialNewsScraper(output_dir=tmpdir)
            path = scraper.save_jsonl(articles, "test_articles.jsonl")

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 5

    def test_save_csv(self):
        articles = [
            NewsArticle(
                article_id="id_001",
                title="BCA laba naik",
                text="BCA membukukan laba bersih yang meningkat signifikan.",
                source="kontan",
                url="https://kontan.co.id/1",
                published_at="2023-06-15",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = FinancialNewsScraper(output_dir=tmpdir)
            path = scraper.save_csv(articles, "test.csv")
            assert path.exists()
            content = path.read_text()
            assert "BCA laba naik" in content

    def test_unknown_source_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = FinancialNewsScraper(output_dir=tmpdir)
            with pytest.raises(ValueError, match="Unknown source"):
                scraper.scrape_source("nonexistent_source")


# ---------------------------------------------------------------------------
# TopicModelResult Tests
# ---------------------------------------------------------------------------

class TestTopicModelResult:

    def _make_result(self) -> TopicModelResult:
        return TopicModelResult(
            model_name="test_model",
            processed_at="2023-01-01T00:00:00",
            n_documents=12,
            n_topics=3,
            embedding_model="test-model",
            topics=[
                TopicInfo(
                    topic_id=0,
                    label="Suku Bunga BI",
                    size=5,
                    representative_words=["suku", "bunga", "bi", "rate"],
                    representative_docs=["BI naikkan suku bunga..."],
                    probability=0.42,
                ),
                TopicInfo(
                    topic_id=1,
                    label="Pasar Saham IHSG",
                    size=4,
                    representative_words=["ihsg", "saham", "bursa"],
                    representative_docs=["IHSG menguat..."],
                    probability=0.33,
                ),
            ],
            topic_assignments=[0, 1, 0, 1, 0, 1, 0, 2, 2, 2, 1, 0],
            outlier_count=0,
        )

    def test_to_json_valid(self):
        result = self._make_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["model_name"] == "test_model"
        assert parsed["n_topics"] == 3
        assert len(parsed["topics"]) == 2

    def test_summary_contains_topic_labels(self):
        result = self._make_result()
        summary = result.summary()
        assert "Suku Bunga BI" in summary
        assert "Pasar Saham IHSG" in summary

    def test_to_dict_serializable(self):
        result = self._make_result()
        d = result.to_dict()
        # Ensure JSON serializable
        json.dumps(d)


# ---------------------------------------------------------------------------
# FinancialTopicModeler Tests
# ---------------------------------------------------------------------------

class TestFinancialTopicModeler:

    def test_raises_on_too_few_documents(self):
        """Should raise ValueError with fewer than 10 documents."""
        import sys
        import types

        # Create fake modules so patch targets exist
        for mod_name in ["bertopic", "sentence_transformers", "umap", "hdbscan"]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)

        import src.nlp.topic_modeler as tm

        # Inject fake classes into module namespace for patching
        tm.BERTopic = MagicMock()
        tm.SentenceTransformer = MagicMock()
        tm.UMAP = MagicMock()
        tm.HDBSCAN = MagicMock()
        tm.HAS_BERTOPIC = True

        modeler = FinancialTopicModeler.__new__(FinancialTopicModeler)
        modeler.embedding_model_name = "test"
        modeler.n_topics = "auto"
        modeler.min_topic_size = 5
        modeler.output_dir = Path("/tmp")
        modeler._embedding_model = None
        modeler._topic_model = None

        with pytest.raises(ValueError, match="at least 10"):
            modeler.fit(["short text"] * 5)

    def test_auto_label_suku_bunga(self):
        """Auto-label should detect interest rate topic."""
        label = FinancialTopicModeler._auto_label(
            ["suku", "bunga", "bi", "rate", "acuan"]
        )
        assert label == "Suku Bunga BI"

    def test_auto_label_inflasi(self):
        label = FinancialTopicModeler._auto_label(
            ["inflasi", "harga", "konsumen", "cpi"]
        )
        assert label == "Inflasi & Harga"

    def test_auto_label_fallback(self):
        """Unknown words should fall back to top-3-words label."""
        label = FinancialTopicModeler._auto_label(
            ["xyz", "abc", "def", "ghi"]
        )
        assert "Xyz" in label or "xyz" in label.lower()

    def test_auto_label_ihsg(self):
        label = FinancialTopicModeler._auto_label(
            ["ihsg", "saham", "bursa", "indeks"]
        )
        assert label == "Pasar Saham IHSG"

    def test_save_result(self):
        """Should save JSON file to output directory."""
        result = TopicModelResult(
            model_name="test",
            processed_at="2023-01-01",
            n_documents=10,
            n_topics=2,
            embedding_model="test",
            topics=[],
            topic_assignments=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            import src.nlp.topic_modeler as tm
            tm.HAS_BERTOPIC = True

            modeler = FinancialTopicModeler.__new__(FinancialTopicModeler)
            modeler.output_dir = Path(tmpdir)
            modeler.embedding_model_name = "test"
            modeler._embedding_model = None
            modeler._topic_model = None

            path = modeler.save(result, "topics_test.json")
            assert path.exists()
            saved = json.loads(path.read_text())
            assert saved["model_name"] == "test"
