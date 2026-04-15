"""
FinSight IDX — Financial Tools with Claude Tool Use
=====================================================
Implements Claude tool use (function calling) for financial analysis.

Tools:
    1. hitung_rasio_keuangan  — compute PER, PBV, ROE, DER, NPM, ROA
    2. bandingkan_emiten      — side-by-side metric comparison
    3. cari_di_laporan        — RAG retrieval from indexed documents

Architecture follows Anthropic tool use pattern:
    User message
        → Claude decides which tool to call
        → stop_reason = "tool_use"
        → application executes tool
        → result fed back as tool_result
        → Claude generates final answer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.api.client import get_client

# RAG retriever (optional — gracefully disabled if not indexed)
try:
    from src.rag.retriever import RAGRetriever
    HAS_RAG = True
except ImportError:
    HAS_RAG = False


# ---------------------------------------------------------------------------
# Tool Definitions (Anthropic JSON Schema format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "hitung_rasio_keuangan",
        "description": (
            "Menghitung rasio keuangan fundamental dari data yang diberikan. "
            "Mendukung: PER (Price-to-Earnings), PBV (Price-to-Book Value), "
            "ROE (Return on Equity), DER (Debt-to-Equity), NPM (Net Profit Margin), "
            "ROA (Return on Assets). "
            "Gunakan tool ini ketika user meminta perhitungan atau analisis rasio keuangan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "emiten": {
                    "type": "string",
                    "description": "Kode saham emiten, contoh: BBCA, BBRI, BMRI",
                },
                "harga_saham": {
                    "type": "number",
                    "description": "Harga saham saat ini dalam Rupiah",
                },
                "eps": {
                    "type": "number",
                    "description": "Earnings Per Share (laba per saham) dalam Rupiah",
                },
                "book_value_per_share": {
                    "type": "number",
                    "description": "Book value per saham dalam Rupiah",
                },
                "laba_bersih": {
                    "type": "number",
                    "description": "Laba bersih dalam miliar Rupiah",
                },
                "ekuitas": {
                    "type": "number",
                    "description": "Total ekuitas dalam miliar Rupiah",
                },
                "total_hutang": {
                    "type": "number",
                    "description": "Total hutang dalam miliar Rupiah",
                },
                "total_aset": {
                    "type": "number",
                    "description": "Total aset dalam miliar Rupiah",
                },
                "pendapatan": {
                    "type": "number",
                    "description": "Total pendapatan/revenue dalam miliar Rupiah",
                },
            },
            "required": ["emiten"],
        },
    },
    {
        "name": "bandingkan_emiten",
        "description": (
            "Membandingkan metrik keuangan antara dua emiten secara berdampingan. "
            "Menghasilkan tabel perbandingan lengkap dengan rekomendasi berdasarkan "
            "rasio keuangan. Gunakan tool ini ketika user ingin membandingkan dua saham."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "emiten_a": {
                    "type": "object",
                    "description": "Data keuangan emiten pertama",
                    "properties": {
                        "kode": {"type": "string"},
                        "nama": {"type": "string"},
                        "harga_saham": {"type": "number"},
                        "eps": {"type": "number"},
                        "book_value_per_share": {"type": "number"},
                        "roe": {"type": "number"},
                        "der": {"type": "number"},
                        "npm": {"type": "number"},
                        "pertumbuhan_laba": {"type": "number"},
                    },
                    "required": ["kode"],
                },
                "emiten_b": {
                    "type": "object",
                    "description": "Data keuangan emiten kedua",
                    "properties": {
                        "kode": {"type": "string"},
                        "nama": {"type": "string"},
                        "harga_saham": {"type": "number"},
                        "eps": {"type": "number"},
                        "book_value_per_share": {"type": "number"},
                        "roe": {"type": "number"},
                        "der": {"type": "number"},
                        "npm": {"type": "number"},
                        "pertumbuhan_laba": {"type": "number"},
                    },
                    "required": ["kode"],
                },
            },
            "required": ["emiten_a", "emiten_b"],
        },
    },
    {
        "name": "cari_di_laporan",
        "description": (
            "Mencari informasi spesifik dari laporan tahunan IDX yang sudah diindeks. "
            "Mengembalikan kutipan relevan dengan nomor halaman sebagai referensi. "
            "Gunakan tool ini untuk pertanyaan faktual tentang isi laporan keuangan, "
            "seperti angka spesifik, kebijakan, atau penjelasan manajemen."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Pertanyaan atau topik yang dicari dalam laporan",
                },
                "emiten": {
                    "type": "string",
                    "description": "Kode saham emiten untuk filter pencarian, contoh: BBCA",
                },
                "tahun": {
                    "type": "integer",
                    "description": "Tahun laporan untuk filter pencarian, contoh: 2023",
                },
                "tipe_seksi": {
                    "type": "string",
                    "description": "Tipe seksi laporan: financial, risk, outlook, governance",
                    "enum": ["financial", "risk", "outlook", "governance", "general"],
                },
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool Implementation Functions
# ---------------------------------------------------------------------------

def hitung_rasio_keuangan(
    emiten: str,
    harga_saham: Optional[float] = None,
    eps: Optional[float] = None,
    book_value_per_share: Optional[float] = None,
    laba_bersih: Optional[float] = None,
    ekuitas: Optional[float] = None,
    total_hutang: Optional[float] = None,
    total_aset: Optional[float] = None,
    pendapatan: Optional[float] = None,
) -> dict:
    """
    Calculate financial ratios from provided data.

    Returns dict with computed ratios and interpretation.
    """
    result: dict[str, Any] = {
        "emiten": emiten.upper(),
        "rasio": {},
        "interpretasi": [],
        "catatan": [],
    }

    # PER — Price to Earnings Ratio
    if harga_saham and eps and eps > 0:
        per = round(harga_saham / eps, 2)
        result["rasio"]["PER"] = per
        if per < 10:
            result["interpretasi"].append(f"PER {per}x — undervalued (< 10x)")
        elif per <= 20:
            result["interpretasi"].append(f"PER {per}x — fairly valued (10-20x)")
        else:
            result["interpretasi"].append(f"PER {per}x — premium/overvalued (> 20x)")

    # PBV — Price to Book Value
    if harga_saham and book_value_per_share and book_value_per_share > 0:
        pbv = round(harga_saham / book_value_per_share, 2)
        result["rasio"]["PBV"] = pbv
        if pbv < 1:
            result["interpretasi"].append(f"PBV {pbv}x — diperdagangkan di bawah book value")
        elif pbv <= 3:
            result["interpretasi"].append(f"PBV {pbv}x — wajar untuk bank Indonesia")
        else:
            result["interpretasi"].append(f"PBV {pbv}x — premium tinggi")

    # ROE — Return on Equity
    if laba_bersih and ekuitas and ekuitas > 0:
        roe = round((laba_bersih / ekuitas) * 100, 2)
        result["rasio"]["ROE"] = f"{roe}%"
        if roe >= 20:
            result["interpretasi"].append(f"ROE {roe}% — excellent (≥ 20%)")
        elif roe >= 15:
            result["interpretasi"].append(f"ROE {roe}% — good (15-20%)")
        else:
            result["interpretasi"].append(f"ROE {roe}% — perlu perhatian (< 15%)")

    # DER — Debt to Equity Ratio
    if total_hutang and ekuitas and ekuitas > 0:
        der = round(total_hutang / ekuitas, 2)
        result["rasio"]["DER"] = f"{der}x"
        if der < 1:
            result["interpretasi"].append(f"DER {der}x — leverage rendah (sehat)")
        elif der <= 3:
            result["interpretasi"].append(f"DER {der}x — leverage moderat")
        else:
            result["interpretasi"].append(f"DER {der}x — leverage tinggi (perhatian)")

    # NPM — Net Profit Margin
    if laba_bersih and pendapatan and pendapatan > 0:
        npm = round((laba_bersih / pendapatan) * 100, 2)
        result["rasio"]["NPM"] = f"{npm}%"
        result["interpretasi"].append(f"NPM {npm}% — margin laba bersih")

    # ROA — Return on Assets
    if laba_bersih and total_aset and total_aset > 0:
        roa = round((laba_bersih / total_aset) * 100, 2)
        result["rasio"]["ROA"] = f"{roa}%"
        if roa >= 2:
            result["interpretasi"].append(f"ROA {roa}% — efisiensi aset baik")
        else:
            result["interpretasi"].append(f"ROA {roa}% — efisiensi aset perlu ditingkatkan")

    if not result["rasio"]:
        result["catatan"].append(
            "Data tidak cukup untuk menghitung rasio. "
            "Berikan minimal: harga_saham + eps untuk PER, "
            "atau laba_bersih + ekuitas untuk ROE."
        )

    return result


def bandingkan_emiten(emiten_a: dict, emiten_b: dict) -> dict:
    """
    Compare two emitens side-by-side with scoring.

    Returns comparison table and recommendation.
    """
    kode_a = emiten_a.get("kode", "A").upper()
    kode_b = emiten_b.get("kode", "B").upper()

    metrics = ["harga_saham", "eps", "book_value_per_share",
               "roe", "der", "npm", "pertumbuhan_laba"]
    labels = {
        "harga_saham": "Harga Saham (Rp)",
        "eps": "EPS (Rp)",
        "book_value_per_share": "Book Value/Share (Rp)",
        "roe": "ROE (%)",
        "der": "DER (x)",
        "npm": "NPM (%)",
        "pertumbuhan_laba": "Pertumbuhan Laba (%)",
    }

    comparison: list[dict] = []
    score_a = score_b = 0

    for metric in metrics:
        val_a = emiten_a.get(metric)
        val_b = emiten_b.get(metric)

        if val_a is None and val_b is None:
            continue

        row = {
            "metrik": labels.get(metric, metric),
            kode_a: val_a if val_a is not None else "N/A",
            kode_b: val_b if val_b is not None else "N/A",
            "lebih_baik": "N/A",
        }

        if val_a is not None and val_b is not None:
            # Higher is better for: ROE, NPM, EPS, pertumbuhan_laba
            # Lower is better for: DER, PER
            higher_better = metric in {"roe", "npm", "eps", "pertumbuhan_laba"}
            lower_better = metric in {"der"}

            if higher_better:
                if val_a > val_b:
                    row["lebih_baik"] = kode_a
                    score_a += 1
                elif val_b > val_a:
                    row["lebih_baik"] = kode_b
                    score_b += 1
                else:
                    row["lebih_baik"] = "Sama"
            elif lower_better:
                if val_a < val_b:
                    row["lebih_baik"] = kode_a
                    score_a += 1
                elif val_b < val_a:
                    row["lebih_baik"] = kode_b
                    score_b += 1
                else:
                    row["lebih_baik"] = "Sama"

        comparison.append(row)

    # Derive recommendation
    if score_a > score_b:
        rekomendasi = f"{kode_a} unggul ({score_a} vs {score_b} metrik)"
    elif score_b > score_a:
        rekomendasi = f"{kode_b} unggul ({score_b} vs {score_a} metrik)"
    else:
        rekomendasi = f"Seimbang ({kode_a} = {kode_b}, pertimbangkan faktor lain)"

    return {
        "perbandingan": comparison,
        "skor": {kode_a: score_a, kode_b: score_b},
        "rekomendasi": rekomendasi,
        "catatan": "Analisis berdasarkan data yang diberikan. Lakukan due diligence lebih lanjut.",
    }


def cari_di_laporan(
    query: str,
    emiten: Optional[str] = None,
    tahun: Optional[int] = None,
    tipe_seksi: Optional[str] = None,
    persist_dir: str = "data/vectorstore",
) -> dict:
    """
    Search indexed IDX annual reports using RAG retriever.

    Returns relevant excerpts with page citations.
    """
    if not HAS_RAG:
        return {
            "error": "RAG retriever tidak tersedia.",
            "query": query,
            "hasil": [],
        }

    try:
        retriever = RAGRetriever(persist_dir=persist_dir)
        chunks = retriever.retrieve(
            query=query,
            top_k=4,
            company_filter=emiten,
            year_filter=tahun,
            section_type_filter=tipe_seksi,
        )

        if not chunks:
            return {
                "query": query,
                "hasil": [],
                "pesan": (
                    "Tidak ditemukan hasil. Pastikan dokumen sudah diindeks "
                    "dengan RAGIndexer.index_pdf() terlebih dahulu."
                ),
            }

        return {
            "query": query,
            "filter": {
                "emiten": emiten,
                "tahun": tahun,
                "tipe_seksi": tipe_seksi,
            },
            "hasil": [
                {
                    "kutipan": chunk.text[:400],
                    "sumber": chunk.citation(),
                    "halaman": chunk.page_number,
                    "seksi": chunk.section_title,
                    "relevansi": chunk.rrf_score,
                }
                for chunk in chunks
            ],
        }

    except ValueError as exc:
        # Collection empty
        return {
            "query": query,
            "hasil": [],
            "pesan": str(exc),
        }
    except Exception as exc:
        logger.error("RAG search error: {}", exc)
        return {
            "query": query,
            "hasil": [],
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Tool Dispatcher
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "hitung_rasio_keuangan": hitung_rasio_keuangan,
    "bandingkan_emiten": bandingkan_emiten,
    "cari_di_laporan": cari_di_laporan,
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool by name with given inputs.

    Args:
        tool_name:  Name of the tool to execute.
        tool_input: Dict of tool arguments from Claude.

    Returns:
        JSON string of tool result.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({
            "error": f"Tool '{tool_name}' tidak dikenal.",
            "available_tools": list(TOOL_REGISTRY.keys()),
        }, ensure_ascii=False)

    try:
        func = TOOL_REGISTRY[tool_name]
        result = func(**tool_input)
        logger.debug("Tool '{}' executed successfully", tool_name)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except TypeError as exc:
        return json.dumps({
            "error": f"Parameter tidak valid untuk tool '{tool_name}': {exc}"
        }, ensure_ascii=False)
    except Exception as exc:
        logger.error("Tool '{}' failed: {}", tool_name, exc)
        return json.dumps({
            "error": f"Tool '{tool_name}' gagal: {str(exc)}"
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in the multi-turn conversation."""
    role: str
    content: Any  # str or list of content blocks


@dataclass
class AnalystResponse:
    """Response from the financial analyst agent."""
    answer: str
    tools_used: list[str]
    tool_results: list[dict]
    conversation_history: list[dict]
    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0


# ---------------------------------------------------------------------------
# Financial Analyst Agent
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT = """Anda adalah analis keuangan senior yang ahli di pasar modal Indonesia (IDX/BEI).

Kemampuan Anda:
- Menghitung dan menginterpretasi rasio keuangan (PER, PBV, ROE, DER, NPM, ROA)
- Membandingkan fundamental antar emiten secara objektif
- Mencari data spesifik dari laporan tahunan perusahaan IDX

Panduan penggunaan tools:
1. Gunakan `hitung_rasio_keuangan` untuk kalkulasi dari data yang diberikan user
2. Gunakan `bandingkan_emiten` untuk perbandingan dua saham
3. Gunakan `cari_di_laporan` untuk mencari fakta dari laporan tahunan

Gaya komunikasi:
- Formal dan profesional, tapi mudah dipahami
- Sertakan konteks dan interpretasi, bukan hanya angka
- Tambahkan disclaimer bahwa analisis bukan rekomendasi investasi
- Jawab dalam Bahasa Indonesia"""


class FinancialAnalystAgent:
    """
    Multi-turn conversational financial analyst powered by Claude tool use.

    Implements the agentic loop:
        User message
            → Claude (with tools)
            → if tool_use: execute tool, feed result back
            → repeat until stop_reason = "end_turn"
            → return final answer

    Usage:
        agent = FinancialAnalystAgent()
        response = agent.chat("Hitung PER BBCA jika harga Rp 9500, EPS Rp 485")
        print(response.answer)

        # Multi-turn — continue conversation
        response2 = agent.chat(
            "Bandingkan dengan BBRI yang harga Rp 4800, EPS Rp 320",
            history=response.conversation_history
        )
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tool_rounds: int = 5,
        rag_persist_dir: str = "data/vectorstore",
    ) -> None:
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.rag_persist_dir = rag_persist_dir

    def chat(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
    ) -> AnalystResponse:
        """
        Send a message to the financial analyst and get a response.

        Automatically handles tool calls in an agentic loop.

        Args:
            user_message: User's question or request.
            history:      Previous conversation history (for multi-turn).

        Returns:
            AnalystResponse with answer, tools used, and updated history.
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})

        tools_used: list[str] = []
        tool_results_log: list[dict] = []
        total_input_tokens = total_output_tokens = 0
        turns = 0

        logger.info("Agent chat | message='{}'", user_message[:60])

        # Agentic loop
        while turns < self.max_tool_rounds:
            turns += 1

            response = get_client().messages.create(
                model=self.model,
                max_tokens=2048,
                system=ANALYST_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Add assistant response to history
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract final text answer
                answer = " ".join(
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ).strip()
                break

            if response.stop_reason == "tool_use":
                # Execute all tool calls in this response
                tool_result_content = []

                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input
                    tools_used.append(tool_name)

                    logger.info(
                        "Executing tool: {} | input_keys={}",
                        tool_name,
                        list(tool_input.keys()),
                    )

                    # Inject RAG persist_dir if needed
                    if tool_name == "cari_di_laporan":
                        tool_input = {
                            **tool_input,
                            "persist_dir": self.rag_persist_dir,
                        }

                    result_str = execute_tool(tool_name, tool_input)
                    result_data = json.loads(result_str)

                    tool_results_log.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "result": result_data,
                    })

                    tool_result_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

                # Feed tool results back to Claude
                messages.append({
                    "role": "user",
                    "content": tool_result_content,
                })
                continue

            # Unexpected stop reason
            logger.warning("Unexpected stop_reason: {}", response.stop_reason)
            answer = "Terjadi kesalahan dalam pemrosesan."
            break
        else:
            answer = "Batas maksimum tool calls tercapai."
            logger.warning("Max tool rounds ({}) reached", self.max_tool_rounds)

        logger.success(
            "Agent complete | tools={} | turns={} | tokens={}/{}",
            tools_used, turns, total_input_tokens, total_output_tokens
        )

        return AnalystResponse(
            answer=answer,
            tools_used=tools_used,
            tool_results=tool_results_log,
            conversation_history=messages,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            turns=turns,
        )
