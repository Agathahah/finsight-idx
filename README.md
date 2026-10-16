# FinSight IDX 🔍📊

> **Financial NLP Intelligence Platform** — AI-powered analysis of IDX annual reports
> and Indonesian financial news using Claude API, RAG pipelines, and MCP server integration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Claude API](https://img.shields.io/badge/Claude_API-Anthropic-D97706)](https://www.anthropic.com/)
[![FastMCP](https://img.shields.io/badge/FastMCP-MCP_Server-6366F1)](https://gofastmcp.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-22C55E)](https://docs.trychroma.com/)
[![Tests](https://img.shields.io/badge/unit_tests-206-22C55E)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Agathahah/finsight-idx/blob/main/notebooks/demo_finsight.ipynb)

---

## 🎯 Project Overview

FinSight IDX is an **end-to-end NLP intelligence platform** for Indonesian capital markets (IDX/BEI). It covers the full workflow from raw PDF ingestion to an MCP-connected analyst agent.

**Status: research prototype.** It runs locally and in a Gradio demo; it has no authentication, monitoring or load testing, and its retrieval quality has not yet been measured on a labelled question set (see *Audit*).

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     FinSight IDX Pipeline                        │
│                                                                  │
│  [Input]  PDF Laporan Tahunan IDX (747 halaman)                  │
│      │                                                           │
│      ▼                                                           │
│  PDFExtractor ──► 423 sections ──► DocumentSummarizer            │
│                                         │                        │
│                                   Claude API                     │
│                              ┌──────────┴──────────┐            │
│                         Summarize            Key Metrics         │
│                         Risk Factors         Outlook             │
│                              │                                   │
│         ┌────────────────────┼──────────────────────┐           │
│         ▼                    ▼                       ▼           │
│    ChromaDB              BERTopic              Sentiment         │
│   (re-index needed)     Topic Modeling        News vs Report      │
│   (RAG Index)          (12 categories)       (BI benchmark)      │
│         │                                                        │
│         ▼                                                        │
│   FinancialQAChain  ◄──  RAGRetriever                           │
│   Claude + Citations      BM25 + Semantic + RRF                  │
│         │                                                        │
│         ▼                                                        │
│   FinancialAnalystAgent  (Claude Tool Use)                       │
│   hitung_rasio │ bandingkan_emiten │ cari_di_laporan             │
│         │                                                        │
│         ▼                                                        │
│   MCP Server (FastMCP) ◄──► Claude Desktop                       │
│   FinSightOrchestrator ──► Markdown Report                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Module | Description | Tech |
|--------|-------------|------|
| 📄 **PDF Extraction** | Section-aware extraction from 747-page IDX reports | pdfplumber |
| 🤖 **Summarization** | Multi-task NLP: summarize, sentiment, metrics, risk | Claude API |
| 🏷️ **Topic Modeling** | Neural topic detection on Indonesian financial news | BERTopic |
| 🔍 **RAG Pipeline** | Hybrid BM25+semantic search with page citations | ChromaDB |
| 📊 **Evaluation** | ROUGE + Claude-as-judge (4.40/5.00 score) | rouge-score |
| 🧮 **Tool Use** | PER/PBV/ROE/DER calculator via Claude tool use | Anthropic SDK |
| 🛠️ **MCP Server** | 4 tools + 2 prompts + 1 resource for Claude Desktop | FastMCP |
| 🤖 **Agent** | 5-step orchestrator: PDF→topics→sentiment→RAG→report | Multi-turn |

---

## 📊 Evaluation Results

### Automatic Metrics (ROUGE)

| Metric | Score |
|--------|-------|
| ROUGE-1 F1 | 0.239 |
| ROUGE-2 F1 | 0.105 |
| ROUGE-L F1 | 0.158 |

### Model-Based Grading (Claude-as-Judge, scale 1–5)

| Criterion | Score |
|-----------|-------|
| Factual Accuracy | 5.00 |
| Financial Relevance | 5.00 |
| Completeness | 4.00 |
| Conciseness | 2.33 |
| **Overall** | **4.40** |

> **How much these numbers can carry:** they come from 10 summaries. The judge is a Claude model grading Claude output, which tends to be lenient toward its own style, and no human rating was collected to check it. Treat 4.40/5 as a smoke test, not as evidence of quality. Low ROUGE is expected for paraphrasing summaries, so ROUGE alone says little either way.

### RAG evaluation (hand-labelled golden set)

Two harnesses, both run against the real index and a golden set whose pages and figures were checked in the PDF by hand (PDF page numbers, not the numbers printed in the footer):

| Harness | Question it answers | Command |
|---|---|---|
| `src/rag/eval_retrieval.py` | Does the right page reach the model? recall@k, MRR and context hit for hybrid, semantic-only and BM25-only | `python -m src.rag.eval_retrieval eval/golden_questions.jsonl --k 10` |
| `src/rag/eval_answers.py` | Is the answer's figure right, is the cited page right, does it refuse when the report has no answer? (paid: ~$0.01 per question) | `python -m src.rag.eval_answers eval/golden_questions.jsonl --limit 58 --top-k 10 --retrieval-mode bm25` |

Golden set: 58 questions (53 answerable, 5 that must be refused) in six groups: financial highlights, ratios, management narrative, governance, financial statements and scanned pages. Questions were drafted with Claude from the PDF text; every figure and page was checked by the author against the PDF and the index (`eval/golden_questions.jsonl`).

**Retrieval.** 53 answerable questions, 95% Wilson intervals in brackets. Each index version was re-built from the PDF and measured on the same questions:

| Index | Retriever | k | recall@k (page) | MRR | context hit@k |
|---|---|---|---|---|---|
| v2 (audit chunker) | Hybrid | 5 | 0.30 | 0.13 | – |
| v3 (no gaps, contextual headers) | Hybrid | 5 | 0.40 [0.28, 0.53] | 0.17 | 0.45 [0.33, 0.59] |
| v4 (+ one chunk per table row) | Semantic only | 5 | 0.25 [0.15, 0.38] | 0.11 | 0.34 [0.23, 0.47] |
| v4 | Hybrid | 5 | 0.58 [0.45, 0.71] | 0.27 | 0.64 [0.51, 0.76] |
| v4 | BM25 only | 5 | 0.62 [0.49, 0.74] | 0.38 | 0.68 [0.55, 0.79] |
| v4 | Hybrid | 10 | 0.68 [0.55, 0.79] | 0.28 | 0.70 [0.56, 0.80] |
| **v4** | **BM25 only** | **10** | **0.70** [0.56, 0.80] | **0.39** | **0.81** [0.69, 0.89] |

*Page recall* counts a hit when a retrieved chunk is on a labelled page. *Context hit* counts a hit when the expected figure appears anywhere in the retrieved chunks, whatever the page. The deployed app uses the last row (`RETRIEVAL_MODE=bm25`, `RAG_TOP_K=10`).

What the numbers say:
- **Table rows were the biggest single gain.** Indexing each row of a financial table as its own sentence (`Total Aset: 2023 = 1.408.107; 2022 = 1.314.732 (dalam miliar Rupiah)`) moved hybrid page recall@5 from 0.40 to 0.58. Fixing the chunker (audit row 14) had moved it from 0.30 to 0.40.
- **Embeddings add little on this document.** BM25 alone matches or beats hybrid at both k, and the semantic-only retriever finds a quarter of the pages. The page-level differences between BM25 and hybrid are small (McNemar p = 0.73 at k = 5, p = 1.0 at k = 10), but at k = 10 BM25 puts the expected figure in front of the model for 6 questions hybrid misses and none the other way (p = 0.03). The likely reason: questions name the exact financial term ("NPL bruto", "BOPO"), and MiniLM, trained on general multilingual text with a 128-token limit, separates those terms poorly. Hybrid stays available (`RETRIEVAL_MODE=hybrid`) for paraphrased questions.
- **More context helps.** Going from 5 to 10 chunks with BM25 raises context hit from 0.68 to 0.81 (7 questions gained, none lost, p = 0.02), at roughly double the input tokens per question.
- **What still fails.** Ratios are always found (10/10). Governance questions ("who chairs the audit committee") find the figure but rarely the labelled page (3/10 pages), because names repeat across many profile pages. Scanned pages are found but OCR text rarely matches the expected figure exactly (1/3 context hits).

**Answers.** Same golden set, Claude Sonnet 4.5, scored automatically: a figure counts as correct when its value matches the expected one in any scale (juta, miliar, triliun) within rounding; a citation counts as correct when the cited page is a labelled page or contains the expected figure. The table is generated from `reports/` by `scripts/update_readme_eval.py`.

<!-- answer-table:start -->
| Answer metric | v4 · hybrid · k = 5 | v4 · BM25 · k = 10 (deployed) |
|---|---|---|
| Questions scored | 57 | 58 |
| Answer accuracy (answerable) | 65% | 79% |
| Citation accuracy | 87% | 91% |
| False refusals (answerable, but model said not found) | 21% | 15% |
| Correct refusals (unanswerable) | 100% | 100% |
| Mean latency | 4.5 s | 4.9 s |
| Cost per question | $0.0063 | $0.0094 |
<!-- answer-table:end -->

Next experiments, each measured on the same golden set: (1) re-rank the top 20 with a multilingual cross-encoder; (2) a multilingual embedding model with a longer input limit than MiniLM's 128 tokens, to see whether hybrid can beat BM25; (3) query expansion for governance questions (names, roles).

Raw reports are saved in `reports/`.

## 🔎 Audit, October 2026

I inspected the stored vector index and the code. Findings and fixes:

| # | Finding | Evidence | Fix |
|---|---|---|---|
| 1 | **Chunker produced near-duplicate chunks.** When a sentence boundary fell just after the window start, the window advanced by one character. | Index: 24,875 chunks, only 20,833 distinct texts, average length 88 characters instead of 256; consecutive chunks like "am Kategori 2…", "m Kategori 2…", "Kategori 2…". | Boundary only accepted in the second half of the window; each step advances at least `chunk_size − overlap`. Regression tests in `tests/test_chunker_pages.py`. **Re-index required.** |
| 2 | **Citations pointed to the section's first page,** not the chunk's page. | 26 distinct page numbers across a 747-page report. | Pages are joined with a page-break marker; each chunk's page is computed from its offset. |
| 3 | **"Hybrid" search was re-ranking.** BM25 only scored the 20 chunks already returned by semantic search, so an exact-term match missed by the embeddings could never be retrieved, and BM25 IDF came from 20 documents. | `retriever.py` v1 | BM25 scores the whole metadata-filtered corpus; RRF fuses two independent lists. Test: a lexical-only hit is now returned. |
| 4 | **Metadata pre-filtering had nothing to filter.** | The index holds one company (BBCA) and one year (2023). | Kept, but the benefit can only be shown after indexing several companies/years. |
| 5 | **Model ID hard-coded in 9 places;** the dated ID returned 404 once retired. Unit tests passed because every API call is mocked. | `grep claude-sonnet-4-20250514` | One setting, `CLAUDE_MODEL` in `.env` (`src/config.py`); `scripts/check_model.py` verifies the ID against the API. |
| 6 | **Evaluation overstated** (10 samples, self-grading judge). | Section above | Caveat added; retrieval evaluation harness added. |
| 7 | **`force_reindex=True` crashed** with `NameError`: the existing-chunk lookup only ran when re-indexing was off. | `indexer.py` v1 | Lookup runs in both modes; test added. |
| 8 | README said MIT license but the repository had no LICENSE file; author line said "M.Sc. Data Science". | — | LICENSE added; degree corrected to Master of Mathematics (Data Science). |
| 9 | **Running headers and footers polluted the index.** | After the chunker fix, ~380 chunks contained the navigation bar and ~475 the page footer. | `PDFExtractor` removes lines that repeat at the top or bottom of ≥25% of pages (3 patterns, ~2,100 lines in this report); unit lines such as "(Dalam jutaan Rupiah …)" are kept. |
| 10 | **Scanned pages were unsearchable.** | PDF pages 570–575 (directors' statement and the independent auditor's report) have no text layer. | OCR fallback (tesseract, `FINSIGHT_OCR`); OCR chunks are tagged and cited as `OCR`. Pages that stay empty are listed in the index manifest. |
| 11 | **The deployed demo could not tell it was serving the broken index.** | The Space still held the v1 vector store. | `index_manifest.json` records schema version, embedding model, chunking and PDF hash; the app refuses an incompatible index (`FINSIGHT_STRICT_INDEX=1`). |
| 12 | **All visitors of the Space shared one agent conversation.** | `app.py` v1 kept the history in a module-level list. | History lives in a per-session `gr.State` and is trimmed to `MAX_AGENT_MESSAGES`. |
| 13 | **No guardrails on a paid API.** No input limits, no rate limit, raw exceptions shown to users, a new QA chain (embedding model + BM25 corpus) built for every question. | `app.py` v1 | `src/serving/`: input validation, per-visitor and global rate limits, one cached chain, generic errors with a request id, one JSON log line per request. |
| 14 | **The audit's own chunker fix skipped text and split table rows.** Every step was forced to advance 192 characters even when the cut came earlier, so the text in between was never indexed; cuts also fell mid-number in tables. Found by the first golden-set run (page recall@5 0.30). | 1.8% of the report's characters were in no chunk; chunks such as "as Induk 48.639 …" had lost their row label. | Line-aware chunker with no gaps (test `test_no_text_is_skipped`), and each chunk prefixed with section, page, unit and year columns, e.g. `[Ikhtisar Data Keuangan · hal. 18 · dalam miliar Rupiah · kolom 2023 2022 …]`. Index schema v3. |



## 🏭 Production setup

| Concern | What is in place | Where |
|---|---|---|
| Ingestion quality | Header/footer removal, OCR fallback, page-accurate citations | `src/nlp/pdf_extractor.py`, `src/rag/indexer.py` |
| Index versioning | Manifest checked at start-up; mismatched index refused | `src/rag/manifest.py` |
| Evaluation | Retrieval (recall@k, MRR) and answer-level (accuracy, citation, refusal) on a golden set | `src/rag/eval_*.py`, `eval/` |
| Guardrails | Question/ticker/year validation, 5 req/min per visitor, 20 req/min global, daily cost ceiling (`DAILY_BUDGET_USD`, reset 00:00 UTC), queue limit; the Anthropic Console monthly limit is the hard cap | `src/serving/guard.py`, `app.py` |
| Observability | `logs/queries.jsonl`: status, latency, tokens, cost, cited pages (question text off by default) | `src/serving/telemetry.py` |
| Reproducible deploy | Pinned runtime requirements, Docker image with the embedding model baked in, scripted Space upload | `requirements_spaces.txt`, `Dockerfile`, `scripts/build_space.py` |

Operate it:

```bash
python -m src.serving.telemetry logs/queries.jsonl     # p50/p95 latency, error rate, cost
python scripts/build_space.py                           # dry run: build/space + checks
python scripts/build_space.py --push --set-config --wait   # deploy and wait until the Space runs
python scripts/smoke_test.py Agatha1911/finsight-idx --paid  # guardrails + one real question on the live Space
python scripts/update_readme_eval.py                    # refresh the answer table from reports/
docker build -t finsight-idx . && docker run --rm -p 7860:7860 --env-file .env finsight-idx
```

Known limits: one document is indexed (BBCA 2023), so company/year filters are not yet exercised; OCR text can contain recognition errors and is marked as such; the rate limiter and the daily budget live in process memory (the budget is re-read from today's query log after a restart), which is enough for one Space replica but not for several; the query log on a free Space is lost when the Space restarts.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com/)
- macOS / Linux / WSL2

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Agathahah/finsight-idx.git
cd finsight-idx

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: tambahkan ANTHROPIC_API_KEY=sk-ant-...
```

### Build the index

```bash
# optional, for scanned pages: brew install tesseract tesseract-lang && pip install pytesseract
# put the annual report PDF in data/raw/ first
PYTHONPATH=. python -c "from src.rag.indexer import RAGIndexer; print(RAGIndexer('data/vectorstore').index_pdf('data/raw/bbca_laporan_tahunan_2023.pdf', company='BBCA', year=2023, force_reindex=True))"
cat data/vectorstore/index_manifest.json      # schema version, OCR pages, empty pages
```

### Verify Setup

```bash
python -c "import anthropic; print('✅ SDK:', anthropic.__version__)"
python -c "import chromadb; print('✅ ChromaDB:', chromadb.__version__)"
PYTHONPATH=. pytest tests/ -q  # 206 unit tests (API calls are mocked)
python scripts/check_model.py  # real API call: confirms CLAUDE_MODEL exists for your key
```

---

## 💡 Usage Examples

### 1. Summarize Annual Report

```python
from src.nlp.summarizer import DocumentSummarizer

summarizer = DocumentSummarizer(output_dir='data/processed')
result = summarizer.summarize('data/raw/bbca_laporan_tahunan_2023.pdf')
print(result.financial_highlights)
```

### 2. RAG Q&A with Citations

```python
from src.rag.qa_chain import FinancialQAChain

qa = FinancialQAChain(persist_dir='data/vectorstore')
response = qa.ask("Berapa laba bersih BCA 2023?", company_filter='BBCA')
print(response.format_answer())
```

### 3. Financial Analyst Agent (Tool Use)

```python
from src.api.tools import FinancialAnalystAgent

agent = FinancialAnalystAgent()
response = agent.chat(
    "Hitung PER dan ROE BBCA: harga Rp 9500, EPS Rp 485, "
    "laba bersih Rp 48.600M, ekuitas Rp 210.000M"
)
print(response.answer)
```

### 4. Full Analysis Pipeline

```python
from src.api.agent import FinSightOrchestrator

orchestrator = FinSightOrchestrator()
report = orchestrator.analyze(emiten='BBCA', tahun=2023,
    pdf_path='data/raw/bbca_laporan_tahunan_2023.pdf')
report.save('data/processed')
```

### 5. MCP Server

```bash
# Start server
PYTHONPATH=. python -m src.mcp.server

# claude_desktop_config.json
{
  "mcpServers": {
    "finsight-idx": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/finsight-idx"
    }
  }
}
```

---

## 📁 Project Structure

```
finsight-idx/
├── src/
│   ├── api/
│   │   ├── client.py       # Claude API client + retry logic
│   │   ├── evaluator.py    # ROUGE + Claude-as-judge
│   │   ├── tools.py        # Tool use + FinancialAnalystAgent
│   │   └── agent.py        # FinSightOrchestrator (5-step)
│   ├── nlp/
│   │   ├── pdf_extractor.py    # PDF extraction + section detection
│   │   ├── summarizer.py       # DocumentSummarizer pipeline
│   │   ├── topic_modeler.py    # BERTopic topic modeling
│   │   └── news_scraper.py     # RSS scraper + synthetic dataset
│   ├── rag/
│   │   ├── indexer.py      # ChromaDB indexing + chunking
│   │   ├── retriever.py    # Hybrid BM25+semantic + RRF
│   │   ├── qa_chain.py     # RAG Q&A with citations
│   │   ├── manifest.py     # index_manifest.json: version checks
│   │   ├── eval_retrieval.py   # recall@k, MRR
│   │   └── eval_answers.py     # answer accuracy, citation accuracy, refusals
│   ├── serving/
│   │   ├── guard.py        # input validation, rate limiting
│   │   ├── qa_service.py   # request path used by app.py
│   │   └── telemetry.py    # JSONL query log + summary
│   └── mcp/
│       └── server.py       # FastMCP (4 tools, 2 prompts, 1 resource)
├── scripts/                # check_model, show_page, build_space, smoke_test, update_readme_eval
├── eval/                   # golden sets (hand-labelled)
├── app.py                  # Gradio demo (Hugging Face Space)
├── Dockerfile
├── tests/                  # 206 unit tests (Claude API mocked)
├── notebooks/
│   └── demo_finsight.ipynb # Interactive demo (Colab-ready)
├── data/
│   ├── raw/                # IDX annual report PDFs
│   └── processed/          # Outputs
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

**AI & LLM:** Anthropic Claude API · sentence-transformers · BERTopic

**Storage:** ChromaDB (persistent, cosine HNSW)

**Document Processing:** pdfplumber

**Serving:** FastMCP (MCP Server)

**Quality:** pytest (206 unit tests) · ruff · rouge-score · GitHub Actions

---

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest tests/ -v        # All 206 tests
PYTHONPATH=. pytest tests/ --cov=src  # With coverage
```

---

## 📂 Data Sources

- **[IDX (Bursa Efek Indonesia)](https://www.idx.co.id/)** — Annual reports, publicly available
- **Indonesian financial news** — RSS feeds (Kontan, CNBC Indonesia, Bisnis)

---

## 👤 Author

**Agatha Silalahi** — Data Scientist, Bank Indonesia Institute (BINS)
Master of Mathematics (Data Science), Universitas Indonesia | AI/ML Engineering, Pacmann

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Agatha_Silalahi-0077B5?logo=linkedin)](https://www.linkedin.com/in/agatha-silalahi-722507215/)
[![GitHub](https://img.shields.io/badge/GitHub-Agathahah-181717?logo=github)](https://github.com/Agathahah)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center">
<sub>Built with ❤️ using Claude API · FinSight IDX © 2026 Agatha Silalahi</sub>
</div>
