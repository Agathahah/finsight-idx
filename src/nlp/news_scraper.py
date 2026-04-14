"""
FinSight IDX — Indonesian Financial News Scraper
=================================================
Collects Indonesian financial news articles from legal, public sources:
- RSS feeds (Kontan, Bisnis Indonesia, CNBC Indonesia)
- IDX press releases (publicly available)
- Yahoo Finance Indonesia

All sources are publicly accessible without authentication.
Rate limiting and respectful crawling are enforced by default.
"""

from __future__ import annotations

import csv
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

try:
    import feedparser  # type: ignore
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    logger.warning("feedparser not installed — RSS scraping disabled. "
                   "Run: pip install feedparser")

try:
    from bs4 import BeautifulSoup  # type: ignore
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("beautifulsoup4 not installed — HTML parsing disabled. "
                   "Run: pip install beautifulsoup4 lxml")


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """A single news article with metadata."""
    article_id: str
    title: str
    text: str
    source: str
    url: str
    published_at: str
    category: str = "financial"
    language: str = "id"
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)
        if not self.article_id:
            self.article_id = hashlib.md5(
                self.url.encode()
            ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# RSS Feed Sources (legal, public)
# ---------------------------------------------------------------------------

RSS_FEEDS: dict[str, str] = {
    "kontan":        "https://rss.kontan.co.id/category/finansial",
    "kontan_market": "https://rss.kontan.co.id/category/market",
    "cnbc_market":   "https://www.cnbcindonesia.com/rss/market",
    "cnbc_finance":  "https://www.cnbcindonesia.com/rss/finance",
    "bisnis_market": "https://bisnis.com/feed/rss/finansial/bursa",
    "detik_finance": "https://finance.detik.com/indeks/rss.xml",
    "tempo_bisnis":  "https://rss.tempo.co/bisnis",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; FinSightIDX/1.0; "
        "+https://github.com/Agathahah/finsight-idx)"
    )
}


# ---------------------------------------------------------------------------
# Scraper Class
# ---------------------------------------------------------------------------

class FinancialNewsScraper:
    """
    Scrapes Indonesian financial news from public RSS feeds.

    Usage:
        scraper = FinancialNewsScraper(output_dir="data/processed")
        articles = scraper.scrape_all(max_per_source=100)
        scraper.save(articles, "financial_news_2024.jsonl")
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed",
        request_delay: float = 1.5,
        timeout: float = 10.0,
    ) -> None:
        """
        Args:
            output_dir:     Directory to save scraped articles.
            request_delay:  Seconds between requests (respectful crawling).
            timeout:        HTTP request timeout in seconds.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout
        self._seen_urls: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_all(
        self,
        max_per_source: int = 100,
        sources: Optional[list[str]] = None,
    ) -> list[NewsArticle]:
        """
        Scrape articles from all configured RSS sources.

        Args:
            max_per_source: Maximum articles to collect per source.
            sources:        Specific source keys to scrape. None = all.

        Returns:
            List of NewsArticle objects, deduplicated by URL.
        """
        if not HAS_FEEDPARSER:
            raise ImportError(
                "feedparser required. Run: pip install feedparser"
            )

        feeds = {
            k: v for k, v in RSS_FEEDS.items()
            if sources is None or k in sources
        }

        all_articles: list[NewsArticle] = []

        for source_name, feed_url in feeds.items():
            logger.info("Scraping source: {} | url={}", source_name, feed_url)
            try:
                articles = self._scrape_rss(
                    feed_url, source_name, max_per_source
                )
                all_articles.extend(articles)
                logger.success(
                    "Scraped {} articles from {}", len(articles), source_name
                )
                time.sleep(self.request_delay)
            except Exception as exc:
                logger.error(
                    "Failed to scrape {}: {}", source_name, exc
                )
                continue

        # Deduplicate
        seen: set[str] = set()
        unique = []
        for art in all_articles:
            if art.url not in seen:
                seen.add(art.url)
                unique.append(art)

        logger.success(
            "Total unique articles collected: {}", len(unique)
        )
        return unique

    def scrape_source(
        self,
        source_name: str,
        max_articles: int = 100,
    ) -> list[NewsArticle]:
        """Scrape a single named source."""
        if source_name not in RSS_FEEDS:
            raise ValueError(
                f"Unknown source '{source_name}'. "
                f"Available: {list(RSS_FEEDS.keys())}"
            )
        return self._scrape_rss(
            RSS_FEEDS[source_name], source_name, max_articles
        )

    def save_jsonl(
        self,
        articles: list[NewsArticle],
        filename: str = "financial_news.jsonl",
    ) -> Path:
        """
        Save articles as JSONL (one JSON object per line).
        Format is compatible with BERTopic and HuggingFace datasets.

        Args:
            articles: List of articles to save.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")

        logger.success(
            "Saved {} articles → {}", len(articles), output_path
        )
        return output_path

    def save_csv(
        self,
        articles: list[NewsArticle],
        filename: str = "financial_news.csv",
    ) -> Path:
        """Save articles as CSV for easy inspection in Excel/spreadsheet."""
        output_path = self.output_dir / filename
        if not articles:
            logger.warning("No articles to save.")
            return output_path

        fieldnames = list(asdict(articles[0]).keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(a) for a in articles)

        logger.success(
            "Saved {} articles as CSV → {}", len(articles), output_path
        )
        return output_path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _scrape_rss(
        self,
        feed_url: str,
        source_name: str,
        max_articles: int,
    ) -> list[NewsArticle]:
        """Parse an RSS feed and extract articles."""
        feed = feedparser.parse(feed_url)

        if feed.bozo and not feed.entries:
            logger.warning(
                "Feed parse error for {}: {}", source_name, feed.bozo_exception
            )
            return []

        articles: list[NewsArticle] = []

        for entry in feed.entries[:max_articles]:
            try:
                article = self._entry_to_article(entry, source_name)
                if article and len(article.text) >= 100:
                    articles.append(article)
            except Exception as exc:
                logger.debug("Skipping entry: {}", exc)
                continue

        return articles

    def _entry_to_article(
        self,
        entry: object,
        source_name: str,
    ) -> Optional[NewsArticle]:
        """Convert an RSS feed entry to a NewsArticle."""
        url = getattr(entry, "link", "") or ""
        if not url or url in self._seen_urls:
            return None
        self._seen_urls.add(url)

        title = getattr(entry, "title", "") or ""
        title = self._strip_html(title).strip()

        # Try to get article body from summary/description
        summary = (
            getattr(entry, "summary", "")
            or getattr(entry, "description", "")
            or ""
        )
        text = self._strip_html(summary).strip()

        # If summary too short, try fetching full article
        if len(text) < 200:
            text = self._fetch_article_text(url) or text

        if not text:
            return None

        # Parse date
        published = ""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                published = datetime(
                    *entry.published_parsed[:6],
                    tzinfo=timezone.utc
                ).isoformat()
            except Exception:
                published = datetime.now(timezone.utc).isoformat()
        else:
            published = datetime.now(timezone.utc).isoformat()

        return NewsArticle(
            article_id=hashlib.md5(url.encode()).hexdigest()[:12],
            title=title,
            text=f"{title}. {text}",
            source=source_name,
            url=url,
            published_at=published,
        )

    def _fetch_article_text(self, url: str) -> Optional[str]:
        """
        Fetch and extract main text from an article URL.
        Returns None if fetch fails or BeautifulSoup not available.
        """
        if not HAS_BS4:
            return None

        try:
            time.sleep(self.request_delay)
            response = httpx.get(
                url,
                headers=HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Remove boilerplate tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Try common article body selectors
            for selector in [
                "article",
                '[class*="article-content"]',
                '[class*="post-content"]',
                '[class*="detail-text"]',
                "main",
            ]:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(separator=" ", strip=True)
                    if len(text) > 200:
                        return text[:3000]  # Cap at 3000 chars

            # Fallback: all paragraph text
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            return text[:3000] if len(text) > 200 else None

        except Exception as exc:
            logger.debug("Failed to fetch {}: {}", url[:60], exc)
            return None

    @staticmethod
    def _strip_html(text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        if HAS_BS4:
            return BeautifulSoup(text, "lxml").get_text(separator=" ")
        # Fallback: simple regex
        import re
        return re.sub(r"<[^>]+>", " ", text)


# ---------------------------------------------------------------------------
# Dataset Generator (synthetic fallback for development)
# ---------------------------------------------------------------------------

def generate_sample_dataset(
    n_articles: int = 500,
    output_dir: str | Path = "data/processed",
) -> Path:
    """
    Generate a synthetic Indonesian financial news dataset for development.
    Use this when real scraping is unavailable (CI, offline, etc.).

    The texts are representative of real Indonesian financial news topics:
    inflation, interest rates, stock market, banking, commodities.

    Args:
        n_articles: Number of articles to generate.
        output_dir: Directory to save the dataset.

    Returns:
        Path to the saved JSONL file.
    """
    import random

    TEMPLATES = [
        "Bank Indonesia mempertahankan suku bunga acuan di level {rate}% "
        "dalam Rapat Dewan Gubernur bulan {month}. Keputusan ini sejalan "
        "dengan upaya menjaga stabilitas nilai tukar rupiah.",

        "Inflasi {month} tercatat {inf}% (yoy), {dir} dari bulan sebelumnya. "
        "BPS mencatat kenaikan harga pada kelompok makanan dan energi "
        "sebagai penyumbang utama.",

        "IHSG {move} {pct}% ke level {level} pada perdagangan {day}. "
        "Sektor perbankan dan konsumer menjadi penopang indeks di tengah "
        "tekanan global dari kenaikan yield obligasi AS.",

        "PT {company} Tbk membukukan laba bersih Rp {profit} triliun pada "
        "kuartal {q} 2023, {growth} {pct2}% dibandingkan periode yang sama "
        "tahun lalu. Pertumbuhan kredit menjadi pendorong utama.",

        "Nilai tukar rupiah {rp_move} terhadap dolar AS ke level Rp {rate2} "
        "per USD. Bank Indonesia siap melakukan intervensi untuk menjaga "
        "stabilitas di pasar valuta asing.",

        "OJK mencatat pertumbuhan kredit perbankan sebesar {credit}% (yoy) "
        "pada {month} 2023. Segmen UMKM dan kredit konsumsi menjadi "
        "pendorong utama pertumbuhan.",

        "Harga komoditas {commodity} internasional {c_move} {c_pct}% "
        "mempengaruhi kinerja emiten berbasis sumber daya alam di BEI. "
        "Analis merekomendasikan overweight pada sektor ini.",

        "Cadangan devisa Indonesia per akhir {month} 2023 tercatat "
        "USD {reserve} miliar, {res_move} dari bulan sebelumnya. "
        "Posisi ini setara dengan {months} bulan impor.",

        "Pemerintah menerbitkan Surat Berharga Negara (SBN) senilai "
        "Rp {sbn} triliun dalam lelang pekan ini. Yield SBN 10 tahun "
        "berada di level {yield_}%.",

        "Merger antara {bank1} dan {bank2} resmi selesai, menciptakan "
        "entitas perbankan dengan total aset Rp {asset} triliun. "
        "Regulator telah memberikan persetujuan penuh.",
    ]

    COMPANIES = ["BBCA", "BBRI", "BMRI", "BBNI", "BNGA", "ARTO", "GOTO", "TLKM", "ASII"]
    MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
              "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
    COMMODITIES = ["batubara", "CPO", "nikel", "emas", "minyak mentah"]
    BANKS = ["Bank Mandiri", "BRI", "BNI", "BTN", "CIMB Niaga"]
    DAYS = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat"]

    random.seed(42)
    articles = []

    for i in range(n_articles):
        template = random.choice(TEMPLATES)
        month = random.choice(MONTHS)
        company = random.choice(COMPANIES)

        text = template.format(
            rate=round(random.uniform(5.5, 6.5), 2),
            month=month,
            inf=round(random.uniform(2.5, 5.5), 2),
            dir=random.choice(["naik", "turun", "stabil"]),
            move=random.choice(["menguat", "melemah"]),
            pct=round(random.uniform(0.1, 2.5), 2),
            pct2=round(random.uniform(5, 25), 1),
            level=random.randint(6500, 7500),
            day=random.choice(DAYS),
            company=company,
            profit=round(random.uniform(1, 50), 1),
            q=random.randint(1, 4),
            growth=random.choice(["tumbuh", "turun"]),
            rp_move=random.choice(["menguat", "melemah"]),
            rate2=random.randint(15000, 16500),
            credit=round(random.uniform(5, 15), 1),
            commodity=random.choice(COMMODITIES),
            c_move=random.choice(["naik", "turun"]),
            c_pct=round(random.uniform(1, 10), 1),
            reserve=round(random.uniform(130, 145), 1),
            res_move=random.choice(["naik", "turun"]),
            months=random.randint(6, 8),
            sbn=random.randint(10, 30),
            yield_=round(random.uniform(6.5, 7.5), 2),
            bank1=random.choice(BANKS),
            bank2=random.choice(BANKS),
            asset=random.randint(500, 2000),
        )

        published = datetime(
            2023,
            random.randint(1, 12),
            random.randint(1, 28),
            tzinfo=timezone.utc,
        ).isoformat()

        articles.append(NewsArticle(
            article_id=f"synthetic_{i:04d}",
            title=text[:80],
            text=text,
            source="synthetic",
            url=f"https://example.com/article/{i}",
            published_at=published,
            category="financial",
            language="id",
        ))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "financial_news_synthetic.jsonl"

    with open(save_path, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")

    logger.success(
        "Generated {} synthetic articles → {}", len(articles), save_path
    )
    return save_path
