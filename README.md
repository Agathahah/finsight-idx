# FinSight IDX 🔍📊

> **Financial NLP Intelligence Platform** — AI-powered analysis of IDX annual reports and Indonesian financial news using Claude API, RAG pipelines, and MCP server integration.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Claude API](https://img.shields.io/badge/Claude-Anthropic-orange)](https://www.anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Project Overview

FinSight IDX is an end-to-end **NLP intelligence platform** built for Indonesian capital markets (IDX). It processes PDF annual reports and financial news using a modern AI stack: Claude API for summarization and structured extraction, BERTopic for financial topic modeling, ChromaDB for semantic search, and FastMCP for exposing FinSight tools as a Model Context Protocol server.

**Primary goal:** Production-quality AI/ML engineering portfolio demonstrating the full lifecycle — from raw PDF ingestion to an agentic MCP workflow — validated on real IDX financial data.

---

## ✨ Key Features

| Module | Description |
|--------|-------------|
| 📄 **Document Summarization** | Multi-level summarization of IDX annual reports with Claude API |
| 🏷️ **Topic Modeling** | BERTopic + sentence-transformers on Indonesian financial news |
| 🔍 **RAG Pipeline** | Retrieval-Augmented Generation with ChromaDB vector store & citations |
| 🧮 **Tool Use** | Claude tool-use for financial metric extraction and calculation |
| 🛠️ **MCP Server** | FastMCP server exposing FinSight capabilities as Claude tools |
| 🤖 **Agent Workflow** | End-to-end agentic pipeline for IDX document Q&A |

---

## 🏗️ Architecture

```
finsight-idx/
├── data/
│   ├── raw/              # PDF laporan tahunan IDX (not tracked in git)
│   └── processed/        # Extracted text & metadata (not tracked in git)
├── src/
│   ├── api/              # Claude API client & utilities
│   ├── nlp/              # Summarization, NER, topic modeling
│   ├── rag/              # Chunking, embedding, retrieval pipeline
│   └── mcp/              # FastMCP server & tool definitions
├── notebooks/            # Exploratory analysis & demos
├── tests/                # Unit & integration tests
├── requirements.txt
└── .env.example
```

---

## 🛠️ Tech Stack

**AI & NLP**
- [Anthropic Claude API](https://docs.anthropic.com/) — summarization, extraction, tool use, agent orchestration
- [sentence-transformers](https://www.sbert.net/) — multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- [BERTopic](https://maartengr.github.io/BERTopic/) — neural topic modeling for financial news
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) — NER, classification

**Storage & Retrieval**
- [ChromaDB](https://docs.trychroma.com/) — local vector store for semantic search

**Document Processing**
- [pdfplumber](https://github.com/jsvine/pdfplumber) — structured PDF extraction (tables, text blocks)
- [PyMuPDF](https://pymupdf.readthedocs.io/) — fast rendering & image extraction

**Serving**
- [FastMCP](https://gofastmcp.com/) — MCP server for Claude tool integration

**Quality**
- `black`, `ruff`, `mypy` — formatting, linting, type checking
- `pytest` + `pytest-asyncio` — unit & async tests

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- macOS / Linux / WSL2

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Agathahah/finsight-idx.git
cd finsight-idx

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Verify Setup

```bash
python -c "import anthropic; print('✅ Anthropic SDK:', anthropic.__version__)"
python -c "import chromadb; print('✅ ChromaDB:', chromadb.__version__)"
python -c "import bertopic; print('✅ BERTopic:', bertopic.__version__)"
```

---

## 📓 Development Roadmap

- [x] **[01]** Repository setup & project structure
- [ ] **[02]** Claude API: first call with IDX annual report
- [ ] **[03]** Document summarization pipeline
- [ ] **[04]** Topic modeling on financial news
- [ ] **[05]** Prompt evaluation & grading system
- [ ] **[06]** RAG pipeline with citations
- [ ] **[07]** Tool use: financial metric calculator
- [ ] **[08]** MCP server: FinSight tools
- [ ] **[09]** MCP resources & prompt templates
- [ ] **[10]** End-to-end agent workflow
- [ ] **[11]** Full demo & documentation

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📂 Data

This project uses publicly available documents from:
- **[IDX (Indonesia Stock Exchange)](https://www.idx.co.id/)** — annual reports (Laporan Tahunan) of listed companies
- **Indonesian financial news** — business press releases

Raw PDFs are stored locally under `data/raw/` and are **not tracked** in version control due to file size.

---

## 👤 Author

**Agatha** — Data Scientist Research Assistant, Bank Indonesia Institute (BINS)  
M.Sc. Data Science, Universitas Indonesia | AI/ML Engineering, Pacmann  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/agatha-silalahi-722507215/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Agathahah)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
