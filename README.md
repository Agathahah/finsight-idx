# FinSight IDX 🔍📊

> **Financial NLP Intelligence Platform** — AI-powered analysis of IDX annual reports
> and Indonesian financial news using Claude API, RAG pipelines, and MCP server integration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Claude API](https://img.shields.io/badge/Claude_API-Anthropic-D97706)](https://www.anthropic.com/)
[![FastMCP](https://img.shields.io/badge/FastMCP-MCP_Server-6366F1)](https://gofastmcp.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-22C55E)](https://docs.trychroma.com/)
[![Tests](https://img.shields.io/badge/Tests-143_passed-22C55E)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Agathahah/finsight-idx/blob/main/notebooks/demo_finsight.ipynb)

---

## 🎯 Project Overview

FinSight IDX is an **end-to-end NLP intelligence platform** for Indonesian capital markets (IDX/BEI). It automates the full financial analysis workflow — from raw PDF ingestion to an agentic MCP-powered analyst — using a production-grade AI stack.

**Built as a validated AI/ML engineering portfolio** demonstrating real-world integration of LLM APIs, vector databases, neural topic modeling, and Model Context Protocol.

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
│   24.8K chunks          Topic Modeling        News vs Report      │
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

> ROUGE rendah pada summarization adalah normal — model melakukan parafrase bukan copy-paste. Claude-as-Judge 4.40/5.00 menunjukkan kualitas konten yang tinggi.

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

### Verify Setup

```bash
python -c "import anthropic; print('✅ SDK:', anthropic.__version__)"
python -c "import chromadb; print('✅ ChromaDB:', chromadb.__version__)"
PYTHONPATH=. pytest tests/ -q  # 143 tests should pass
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
│   │   └── qa_chain.py     # RAG Q&A with citations
│   └── mcp/
│       └── server.py       # FastMCP (4 tools, 2 prompts, 1 resource)
├── tests/                  # 143 unit tests (all passing)
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

**Storage:** ChromaDB (24.8K chunks indexed)

**Document Processing:** pdfplumber

**Serving:** FastMCP (MCP Server)

**Quality:** pytest (143 tests) · black · ruff · mypy · rouge-score

---

## 🧪 Running Tests

```bash
PYTHONPATH=. pytest tests/ -v        # All 143 tests
PYTHONPATH=. pytest tests/ --cov=src  # With coverage
```

---

## 📂 Data Sources

- **[IDX (Bursa Efek Indonesia)](https://www.idx.co.id/)** — Annual reports, publicly available
- **Indonesian financial news** — RSS feeds (Kontan, CNBC Indonesia, Bisnis)

---

## 👤 Author

**Agatha Silalahi** — Data Scientist, Bank Indonesia Institute (BINS)
M.Sc. Data Science, Universitas Indonesia | AI/ML Engineering, Pacmann

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Agatha_Silalahi-0077B5?logo=linkedin)](https://www.linkedin.com/in/agatha-silalahi-722507215/)
[![GitHub](https://img.shields.io/badge/GitHub-Agathahah-181717?logo=github)](https://github.com/Agathahah)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

<div align="center">
<sub>Built with ❤️ using Claude API · FinSight IDX © 2024 Agatha Silalahi</sub>
</div>
