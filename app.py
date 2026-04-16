"""
FinSight IDX — Gradio Demo App
================================
Interactive demo for HuggingFace Spaces deployment.

Features:
- Tab 1: Financial Ratio Calculator (hitung_rasio)
- Tab 2: RAG Q&A — Tanya Laporan (tanya_laporan)
- Tab 3: Financial Analyst Agent (multi-turn chat)
- Tab 4: Emiten Comparison (bandingkan_emiten)

Deploy to HuggingFace Spaces:
    1. Create new Space (Gradio SDK)
    2. Upload this file as app.py
    3. Add ANTHROPIC_API_KEY to Space Secrets
    4. Add requirements.txt
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Check API key ────────────────────────────────────────────────────────────
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
HAS_API = API_KEY.startswith("sk-ant-")

# ── Imports (lazy to handle missing deps gracefully) ─────────────────────────
try:
    from src.api.tools import (
        hitung_rasio_keuangan,
        bandingkan_emiten,
        FinancialAnalystAgent,
    )
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False

try:
    from src.rag.qa_chain import FinancialQAChain
    HAS_RAG = True
except ImportError:
    HAS_RAG = False

# ── Agent singleton ──────────────────────────────────────────────────────────
_agent = None
_agent_history = []

def get_agent():
    global _agent
    if _agent is None and HAS_TOOLS and HAS_API:
        _agent = FinancialAnalystAgent()
    return _agent


# ============================================================================
# Tab 1: Financial Ratio Calculator
# ============================================================================

def calculate_ratios(
    emiten, harga_saham, eps, book_value,
    laba_bersih, ekuitas, total_hutang, total_aset, pendapatan
):
    if not emiten:
        return "⚠️ Masukkan kode emiten terlebih dahulu (contoh: BBCA)"

    def to_float(val):
        try:
            return float(str(val).replace(",", ".")) if val else None
        except Exception:
            return None

    result = hitung_rasio_keuangan(
        emiten=emiten,
        harga_saham=to_float(harga_saham),
        eps=to_float(eps),
        book_value_per_share=to_float(book_value),
        laba_bersih=to_float(laba_bersih),
        ekuitas=to_float(ekuitas),
        total_hutang=to_float(total_hutang),
        total_aset=to_float(total_aset),
        pendapatan=to_float(pendapatan),
    )

    if not result["rasio"]:
        return f"⚠️ {result['catatan'][0] if result['catatan'] else 'Data tidak cukup untuk menghitung rasio.'}"

    lines = [f"## 📊 Rasio Keuangan — {result['emiten']}\n"]
    lines.append("| Rasio | Nilai |")
    lines.append("|-------|-------|")
    for k, v in result["rasio"].items():
        lines.append(f"| **{k}** | {v} |")

    lines.append("\n### 💡 Interpretasi")
    for note in result["interpretasi"]:
        emoji = "✅" if any(w in note for w in ["excellent", "undervalued", "sehat", "baik"]) else \
                "⚠️" if any(w in note for w in ["perhatian", "tinggi", "rendah"]) else "📌"
        lines.append(f"{emoji} {note}")

    lines.append("\n---")
    lines.append("*Disclaimer: Bukan rekomendasi investasi.*")
    return "\n".join(lines)


# ============================================================================
# Tab 2: RAG Q&A
# ============================================================================

def ask_laporan(query, emiten_filter, year_filter):
    if not query.strip():
        return "⚠️ Masukkan pertanyaan terlebih dahulu."

    if not HAS_API:
        return "⚠️ ANTHROPIC_API_KEY belum diset. Tambahkan di Space Secrets."

    if not HAS_RAG:
        return "⚠️ RAG module tidak tersedia."

    try:
        qa = FinancialQAChain(persist_dir="data/vectorstore")
        response = qa.ask(
            question=query,
            company_filter=emiten_filter if emiten_filter else None,
            year_filter=int(year_filter) if year_filter else None,
        )

        lines = [f"## 💡 Jawaban\n{response.answer}\n"]

        if response.sources:
            lines.append("## 📚 Sumber")
            for src in response.sources[:3]:
                lines.append(f"- **{src.citation}** — hal. {src.page_number}")
                lines.append(f"  > *{src.text_preview[:120]}...*")

        lines.append(f"\n---\n⚡ {response.input_tokens} in / {response.output_tokens} out tokens | {response.latency_ms:.0f}ms")
        return "\n".join(lines)

    except ValueError as e:
        return (
            f"⚠️ {e}\n\n"
            "**Hint:** Pastikan PDF laporan sudah diindeks dengan:\n"
            "```python\nfrom src.rag.indexer import RAGIndexer\n"
            "indexer = RAGIndexer()\n"
            "indexer.index_pdf('data/raw/bbca_laporan_tahunan_2023.pdf',\n"
            "                   company='BBCA', year=2023)\n```"
        )
    except Exception as e:
        return f"❌ Error: {e}"


# ============================================================================
# Tab 3: Financial Analyst Agent (Chat)
# ============================================================================

def chat_with_agent(message, history):
    # Gradio 6: messages format {"role": ..., "content": ...}
    user_msg = {"role": "user", "content": message}

    if not HAS_API:
        return history + [user_msg, {"role": "assistant", "content": "⚠️ ANTHROPIC_API_KEY belum diset."}]

    if not HAS_TOOLS:
        return history + [user_msg, {"role": "assistant", "content": "⚠️ Tools module tidak tersedia."}]

    agent = get_agent()
    if not agent:
        return history + [user_msg, {"role": "assistant", "content": "⚠️ Agent tidak dapat diinisialisasi."}]

    global _agent_history
    try:
        response = agent.chat(message, history=_agent_history)
        _agent_history = response.conversation_history

        answer = response.answer
        if response.tools_used:
            tools = ", ".join(set(response.tools_used))
            answer += f"\n\n---\n🔧 *Tools: {tools} | {response.input_tokens} tokens*"

        return history + [user_msg, {"role": "assistant", "content": answer}]
    except Exception as e:
        return history + [user_msg, {"role": "assistant", "content": f"❌ Error: {e}"}]


def reset_chat():
    global _agent_history
    _agent_history = []
    return []


# ============================================================================
# Tab 4: Compare Emitens
# ============================================================================

def compare_emitens(
    kode_a, roe_a, der_a, npm_a, eps_a, growth_a, harga_a,
    kode_b, roe_b, der_b, npm_b, eps_b, growth_b, harga_b,
):
    def to_f(v):
        try: return float(str(v).replace(",", ".")) if v else None
        except: return None

    if not kode_a or not kode_b:
        return "⚠️ Masukkan kode kedua emiten."

    emiten_a = {"kode": kode_a}
    emiten_b = {"kode": kode_b}
    for key, val in [("roe", roe_a), ("der", der_a), ("npm", npm_a),
                     ("eps", eps_a), ("pertumbuhan_laba", growth_a),
                     ("harga_saham", harga_a)]:
        v = to_f(val)
        if v is not None:
            emiten_a[key] = v
    for key, val in [("roe", roe_b), ("der", der_b), ("npm", npm_b),
                     ("eps", eps_b), ("pertumbuhan_laba", growth_b),
                     ("harga_saham", harga_b)]:
        v = to_f(val)
        if v is not None:
            emiten_b[key] = v

    result = bandingkan_emiten(emiten_a, emiten_b)

    lines = [f"## ⚖️ {kode_a.upper()} vs {kode_b.upper()}\n"]
    lines.append(f"| Metrik | {kode_a.upper()} | {kode_b.upper()} | Lebih Baik |")
    lines.append("|--------|------|------|------------|")

    for row in result["perbandingan"]:
        winner = row["lebih_baik"]
        badge = (
            f"🏆 **{winner}**" if winner not in ("Sama", "N/A")
            else ("🤝 Sama" if winner == "Sama" else "—")
        )
        val_a = row.get(kode_a.upper(), "N/A")
        val_b = row.get(kode_b.upper(), "N/A")
        lines.append(f"| {row['metrik']} | {val_a} | {val_b} | {badge} |")

    lines.append(f"\n### 🎯 Rekomendasi\n**{result['rekomendasi']}**")
    lines.append(f"\n*Skor: {kode_a.upper()} = {result['skor'].get(kode_a.upper(), 0)} | {kode_b.upper()} = {result['skor'].get(kode_b.upper(), 0)} metrik unggul*")
    lines.append("\n---\n*Disclaimer: Bukan rekomendasi investasi.*")
    return "\n".join(lines)


# ============================================================================
# Gradio UI
# ============================================================================

THEME = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
).set(
    body_background_fill="#0f1117",
    body_text_color="#e2e8f0",
    block_background_fill="#1e2330",
    block_border_color="#2d3748",
    block_title_text_color="#00d4aa",
    input_background_fill="#252d3d",
    input_border_color="#3d4d66",
    button_primary_background_fill="#00d4aa",
    button_primary_text_color="#0f1117",
    button_primary_background_fill_hover="#00b894",
)

# Custom CSS for chat readability
CUSTOM_CSS = """
.message.user {
    background: #1a3a2a !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d5a3d !important;
}
.message.bot, .message.assistant {
    background: #1e2330 !important;
    color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important;
}
.message-wrap .message {
    color: #e2e8f0 !important;
}
.message-wrap .message p,
.message-wrap .message li,
.message-wrap .message h1,
.message-wrap .message h2,
.message-wrap .message h3 {
    color: #e2e8f0 !important;
}
.message-wrap .message code {
    background: #252d3d !important;
    color: #00d4aa !important;
}
.message-wrap .message strong {
    color: #ffffff !important;
}
/* User bubble */
.bubble-wrap.user .bubble {
    background: #1a3a2a !important;
    color: #e2e8f0 !important;
}
/* Bot bubble */  
.bubble-wrap.bot .bubble {
    background: #252d3d !important;
    color: #e2e8f0 !important;
}
"""

HEADER = """
<div style="text-align:center; padding: 20px 0 10px;">
  <h1 style="font-size:2.2rem; font-weight:800; color:#00d4aa; margin:0;">
    🔍 FinSight IDX
  </h1>
  <p style="color:#94a3b8; font-size:1rem; margin:8px 0 0;">
    Financial NLP Intelligence Platform for Indonesian Capital Markets
  </p>
  <div style="margin-top:10px; display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
    <a href="https://github.com/Agathahah/finsight-idx" target="_blank"
       style="background:#1e2330; border:1px solid #3d4d66; padding:4px 12px;
              border-radius:20px; color:#00d4aa; text-decoration:none; font-size:0.85rem;">
      📦 GitHub
    </a>
    <span style="background:#1e2330; border:1px solid #3d4d66; padding:4px 12px;
                 border-radius:20px; color:#94a3b8; font-size:0.85rem;">
      🤖 Claude API
    </span>
    <span style="background:#1e2330; border:1px solid #3d4d66; padding:4px 12px;
                 border-radius:20px; color:#94a3b8; font-size:0.85rem;">
      🔍 RAG + BERTopic
    </span>
    <span style="background:#1e2330; border:1px solid #3d4d66; padding:4px 12px;
                 border-radius:20px; color:#94a3b8; font-size:0.85rem;">
      🛠️ FastMCP
    </span>
  </div>
</div>
"""

with gr.Blocks(title="FinSight IDX", css=CUSTOM_CSS) as demo:
    gr.HTML(HEADER)

    with gr.Tabs():

        # ── Tab 1: Ratio Calculator ─────────────────────────────────────────
        with gr.Tab("📊 Hitung Rasio"):
            gr.Markdown(
                "### Kalkulator Rasio Keuangan\n"
                "Masukkan data keuangan emiten untuk menghitung PER, PBV, ROE, DER, NPM, ROA."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    r_emiten = gr.Textbox(label="Kode Emiten *", placeholder="BBCA", max_lines=1)
                    with gr.Row():
                        r_harga = gr.Number(label="Harga Saham (Rp)", value=9500)
                        r_eps   = gr.Number(label="EPS (Rp)", value=485)
                    with gr.Row():
                        r_bvps  = gr.Number(label="Book Value/Share (Rp)", value=2089)
                        r_laba  = gr.Number(label="Laba Bersih (Miliar Rp)", value=48600)
                    with gr.Row():
                        r_ekuitas = gr.Number(label="Ekuitas (Miliar Rp)", value=210000)
                        r_hutang  = gr.Number(label="Total Hutang (Miliar Rp)", value=900000)
                    with gr.Row():
                        r_aset    = gr.Number(label="Total Aset (Miliar Rp)", value=1408000)
                        r_pendapatan = gr.Number(label="Pendapatan (Miliar Rp)", value=149200)

                    gr.Examples(
                        examples=[
                            ["BBCA", 9500, 485, 2089, 48600, 210000, 900000, 1408000, 149200],
                            ["BBRI", 5500, 320, 1850, 60400, 332000, 1200000, 1900000, 213000],
                            ["BMRI", 6200, 410, 2100, 55100, 280000, 1100000, 2174000, 178000],
                        ],
                        inputs=[r_emiten, r_harga, r_eps, r_bvps, r_laba,
                                r_ekuitas, r_hutang, r_aset, r_pendapatan],
                        label="Contoh Data — Klik untuk isi otomatis",
                    )
                    btn_rasio = gr.Button("⚡ Hitung Rasio", variant="primary", size="lg")

                with gr.Column(scale=1):
                    out_rasio = gr.Markdown(
                        value="*Isi data di sebelah kiri, lalu klik Hitung Rasio.*",
                        label="Hasil",
                    )

            btn_rasio.click(
                calculate_ratios,
                inputs=[r_emiten, r_harga, r_eps, r_bvps, r_laba,
                        r_ekuitas, r_hutang, r_aset, r_pendapatan],
                outputs=out_rasio,
            )

        # ── Tab 2: RAG Q&A ──────────────────────────────────────────────────
        with gr.Tab("💬 Tanya Laporan"):
            gr.Markdown(
                "### RAG Q&A — Laporan Tahunan IDX\n"
                "Tanya apapun tentang laporan tahunan yang sudah diindeks. "
                "Jawaban disertai citation nomor halaman."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    rag_query = gr.Textbox(
                        label="Pertanyaan",
                        placeholder="Berapa laba bersih BCA tahun 2023?",
                        lines=2,
                    )
                    with gr.Row():
                        rag_emiten = gr.Textbox(
                            label="Filter Emiten (opsional)",
                            placeholder="BBCA", max_lines=1
                        )
                        rag_year = gr.Textbox(
                            label="Filter Tahun (opsional)",
                            placeholder="2023", max_lines=1
                        )
                    gr.Examples(
                        examples=[
                            ["Berapa laba bersih BCA tahun 2023?", "BBCA", "2023"],
                            ["Apa saja risiko utama yang dihadapi BCA?", "BBCA", "2023"],
                            ["Bagaimana target pertumbuhan kredit BCA 2024?", "BBCA", "2023"],
                            ["Berapa rasio CAR BCA dan apa artinya?", "BBCA", "2023"],
                        ],
                        inputs=[rag_query, rag_emiten, rag_year],
                        label="Contoh Pertanyaan",
                    )
                    btn_rag = gr.Button("🔍 Cari Jawaban", variant="primary", size="lg")

                with gr.Column(scale=2):
                    out_rag = gr.Markdown(
                        value="*Ketik pertanyaan di sebelah kiri, lalu klik Cari Jawaban.*",
                        label="Jawaban + Citations",
                    )

            btn_rag.click(
                ask_laporan,
                inputs=[rag_query, rag_emiten, rag_year],
                outputs=out_rag,
            )

        # ── Tab 3: Agent Chat ───────────────────────────────────────────────
        with gr.Tab("🤖 Analyst Agent"):
            gr.Markdown(
                "### Financial Analyst Agent\n"
                "Multi-turn conversation dengan AI analyst yang secara otomatis "
                "memanggil tools untuk menjawab pertanyaan keuangan kompleks."
            )
            chatbot = gr.Chatbot(
                label="FinSight Financial Analyst",
                height=420,
                avatar_images=(None, "🤖"),
            )
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Hitung PER BBCA: harga Rp 9500, EPS Rp 485...",
                    label="Pesan",
                    scale=4,
                    max_lines=2,
                )
                btn_send = gr.Button("Kirim", variant="primary", scale=1)

            btn_clear = gr.Button("🗑️ Reset Conversation", size="sm")

            gr.Examples(
                examples=[
                    ["Hitung PER dan ROE BBCA: harga Rp 9500, EPS Rp 485, laba bersih Rp 48600 miliar, ekuitas Rp 210000 miliar"],
                    ["Bandingkan BBCA dan BBRI: BBCA ROE 23.5%, DER 4.2x — BBRI ROE 18.2%, DER 5.1x"],
                    ["Analisis valuasi BMRI: harga Rp 6200, EPS Rp 410, book value Rp 2100 per saham"],
                ],
                inputs=chat_input,
                label="Contoh Pertanyaan",
            )

            btn_send.click(
                chat_with_agent,
                inputs=[chat_input, chatbot],
                outputs=chatbot,
            ).then(lambda: "", outputs=chat_input)

            chat_input.submit(
                chat_with_agent,
                inputs=[chat_input, chatbot],
                outputs=chatbot,
            ).then(lambda: "", outputs=chat_input)

            btn_clear.click(reset_chat, outputs=chatbot)

        # ── Tab 4: Compare ──────────────────────────────────────────────────
        with gr.Tab("⚖️ Bandingkan Emiten"):
            gr.Markdown(
                "### Perbandingan Fundamental Dua Emiten\n"
                "Masukkan metrik keuangan dua emiten untuk perbandingan side-by-side."
            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Emiten A")
                    c_kode_a   = gr.Textbox(label="Kode", placeholder="BBCA", max_lines=1)
                    c_roe_a    = gr.Number(label="ROE (%)", value=23.5)
                    c_der_a    = gr.Number(label="DER (x)", value=4.2)
                    c_npm_a    = gr.Number(label="NPM (%)", value=32.6)
                    c_eps_a    = gr.Number(label="EPS (Rp)", value=485)
                    c_growth_a = gr.Number(label="Pertumbuhan Laba (%)", value=19.5)
                    c_harga_a  = gr.Number(label="Harga Saham (Rp)", value=9500)

                with gr.Column():
                    gr.Markdown("#### Emiten B")
                    c_kode_b   = gr.Textbox(label="Kode", placeholder="BBRI", max_lines=1)
                    c_roe_b    = gr.Number(label="ROE (%)", value=18.2)
                    c_der_b    = gr.Number(label="DER (x)", value=5.1)
                    c_npm_b    = gr.Number(label="NPM (%)", value=28.4)
                    c_eps_b    = gr.Number(label="EPS (Rp)", value=320)
                    c_growth_b = gr.Number(label="Pertumbuhan Laba (%)", value=17.5)
                    c_harga_b  = gr.Number(label="Harga Saham (Rp)", value=5500)

            btn_compare = gr.Button("⚡ Bandingkan", variant="primary", size="lg")
            out_compare = gr.Markdown(value="*Isi data kedua emiten, lalu klik Bandingkan.*")

            gr.Examples(
                examples=[
                    ["BBCA", 23.5, 4.2, 32.6, 485, 19.5, 9500,
                     "BBRI", 18.2, 5.1, 28.4, 320, 17.5, 5500],
                    ["BMRI", 19.8, 5.5, 30.9, 410, 33.6, 6200,
                     "BBNI", 13.5, 5.5, 21.3, 280, 14.2, 5400],
                ],
                inputs=[c_kode_a, c_roe_a, c_der_a, c_npm_a, c_eps_a, c_growth_a, c_harga_a,
                        c_kode_b, c_roe_b, c_der_b, c_npm_b, c_eps_b, c_growth_b, c_harga_b],
                label="Contoh Perbandingan",
            )

            btn_compare.click(
                compare_emitens,
                inputs=[c_kode_a, c_roe_a, c_der_a, c_npm_a, c_eps_a, c_growth_a, c_harga_a,
                        c_kode_b, c_roe_b, c_der_b, c_npm_b, c_eps_b, c_growth_b, c_harga_b],
                outputs=out_compare,
            )

    # Footer
    gr.Markdown(
        """
---
<div style="text-align:center; color:#64748b; font-size:0.8rem;">
Built by <a href="https://www.linkedin.com/in/agatha-silalahi-722507215/"
style="color:#00d4aa;">Agatha Silalahi</a> —
Data Scientist, Bank Indonesia Institute |
<a href="https://github.com/Agathahah/finsight-idx" style="color:#00d4aa;">GitHub</a>
· Powered by Claude API · Not financial advice
</div>
        """,
        sanitize_html=False,
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=THEME,
    )
