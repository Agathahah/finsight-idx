"""
FinSight IDX — Summary Quality Evaluator
==========================================
Two-layer evaluation system for Claude-generated financial summaries:

Layer 1 — Automatic metrics (no API cost):
    - ROUGE-1, ROUGE-2, ROUGE-L  (lexical overlap with reference)
    - Length ratio                (output vs reference length)

Layer 2 — Model-based grading (Claude-as-judge):
    - Factual accuracy     (1–5)
    - Financial relevance  (1–5)
    - Completeness         (1–5)
    - Conciseness          (1–5)
    - Overall score        (weighted average)

Pipeline:
    EvalDataset (source_text + reference_summary)
        → generate candidate summary via Claude API
        → compute ROUGE scores
        → Claude-as-judge grades the candidate
        → EvalResult with all metrics
        → JSON report
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.api.client import FinancialTask, analyze_financial_text

try:
    from rouge_score import rouge_scorer  # type: ignore
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    logger.warning("rouge-score not installed. Run: pip install rouge-score")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """Anda adalah evaluator ahli untuk sistem AI keuangan.
Tugas Anda menilai kualitas ringkasan laporan keuangan IDX yang dihasilkan AI.

Berikan penilaian dalam format JSON dengan struktur PERSIS seperti berikut:
{
  "factual_accuracy": <int 1-5>,
  "financial_relevance": <int 1-5>,
  "completeness": <int 1-5>,
  "conciseness": <int 1-5>,
  "reasoning": "<string penjelasan singkat>"
}

Rubrik penilaian (1=sangat buruk, 5=sempurna):
- factual_accuracy   : Apakah angka dan fakta akurat sesuai sumber?
- financial_relevance: Apakah metrik keuangan utama tercakup?
- completeness       : Apakah semua poin penting dari teks sumber ada?
- conciseness        : Apakah ringkasan padat tanpa informasi tidak relevan?

PENTING: Respons HANYA JSON, tanpa teks tambahan apapun."""

JUDGE_USER_TEMPLATE = """Evaluasi ringkasan berikut:

=== TEKS SUMBER ===
{source_text}

=== RINGKASAN YANG DIEVALUASI ===
{candidate_summary}

=== REFERENSI (ground truth) ===
{reference_summary}

Berikan penilaian JSON sesuai rubrik."""

# Weights for overall score calculation
SCORE_WEIGHTS = {
    "factual_accuracy": 0.35,
    "financial_relevance": 0.30,
    "completeness": 0.20,
    "conciseness": 0.15,
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """A single evaluation sample with source text and reference summary."""
    sample_id: str
    source_text: str
    reference_summary: str
    company: str = ""
    year: int = 2023
    section_type: str = "financial"


@dataclass
class RougeScores:
    """ROUGE metric scores for a candidate summary."""
    rouge1_precision: float = 0.0
    rouge1_recall: float = 0.0
    rouge1_f1: float = 0.0
    rouge2_precision: float = 0.0
    rouge2_recall: float = 0.0
    rouge2_f1: float = 0.0
    rougeL_precision: float = 0.0
    rougeL_recall: float = 0.0
    rougeL_f1: float = 0.0
    length_ratio: float = 0.0


@dataclass
class JudgeScores:
    """Claude-as-judge evaluation scores."""
    factual_accuracy: int = 0
    financial_relevance: int = 0
    completeness: int = 0
    conciseness: int = 0
    overall: float = 0.0
    reasoning: str = ""
    raw_response: str = ""


@dataclass
class EvalResult:
    """Full evaluation result for one sample."""
    sample_id: str
    company: str
    year: int
    source_chars: int
    candidate_summary: str
    reference_summary: str
    rouge: RougeScores
    judge: JudgeScores
    eval_model: str
    evaluated_at: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class EvalReport:
    """Aggregated evaluation report across all samples."""
    report_id: str
    created_at: str
    n_samples: int
    eval_model: str

    # Aggregated ROUGE
    avg_rouge1_f1: float = 0.0
    avg_rouge2_f1: float = 0.0
    avg_rougeL_f1: float = 0.0

    # Aggregated judge scores
    avg_factual_accuracy: float = 0.0
    avg_financial_relevance: float = 0.0
    avg_completeness: float = 0.0
    avg_conciseness: float = 0.0
    avg_overall_judge: float = 0.0

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Per-sample results
    results: list[EvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["results"] = [asdict(r) for r in self.results]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def markdown_table(self) -> str:
        """Generate a Markdown table for README/documentation."""
        lines = [
            "## FinSight IDX — Evaluation Results",
            "",
            f"**Model:** `{self.eval_model}` | "
            f"**Samples:** {self.n_samples} | "
            f"**Date:** {self.created_at[:10]}",
            "",
            "### Automatic Metrics (ROUGE)",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| ROUGE-1 F1 | {self.avg_rouge1_f1:.3f} |",
            f"| ROUGE-2 F1 | {self.avg_rouge2_f1:.3f} |",
            f"| ROUGE-L F1 | {self.avg_rougeL_f1:.3f} |",
            "",
            "### Model-Based Grading (Claude-as-Judge, scale 1–5)",
            "",
            "| Criterion | Score |",
            "|-----------|-------|",
            f"| Factual Accuracy    | {self.avg_factual_accuracy:.2f} |",
            f"| Financial Relevance | {self.avg_financial_relevance:.2f} |",
            f"| Completeness        | {self.avg_completeness:.2f} |",
            f"| Conciseness         | {self.avg_conciseness:.2f} |",
            f"| **Overall**         | **{self.avg_overall_judge:.2f}** |",
            "",
            "### Per-Sample Results",
            "",
            "| ID | Company | ROUGE-L | Judge Overall |",
            "|----|---------|---------|---------------|",
        ]
        for r in self.results:
            lines.append(
                f"| {r.sample_id} | {r.company} | "
                f"{r.rouge.rougeL_f1:.3f} | "
                f"{r.judge.overall:.2f} |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in Test Dataset
# ---------------------------------------------------------------------------

def get_builtin_eval_dataset() -> list[EvalSample]:
    """
    10-sample evaluation dataset built from public IDX financial report excerpts.
    Reference summaries are manually written ground truth.

    Returns:
        List of EvalSample objects ready for evaluation.
    """
    return [
        EvalSample(
            sample_id="bbca_2023_financial",
            company="BBCA",
            year=2023,
            section_type="financial",
            source_text=(
                "BCA membukukan laba bersih sebesar Rp 48,6 triliun pada tahun 2023, "
                "meningkat 19,5% dibandingkan tahun sebelumnya Rp 40,7 triliun. "
                "Total aset BCA tumbuh 8,7% menjadi Rp 1.408 triliun. "
                "Kredit yang disalurkan meningkat 13,8% menjadi Rp 793,2 triliun, "
                "didorong oleh kredit korporasi yang tumbuh 16,2% dan kredit UKM 14,5%. "
                "Rasio kecukupan modal (CAR) tercatat 25,9%, jauh di atas minimum regulator 14%."
            ),
            reference_summary=(
                "BCA mencatat pertumbuhan kinerja keuangan yang kuat di 2023: laba bersih "
                "naik 19,5% menjadi Rp 48,6 triliun, total aset tumbuh 8,7% ke Rp 1.408 T, "
                "dan kredit meningkat 13,8% ke Rp 793,2 T. CAR 25,9% jauh di atas regulasi."
            ),
        ),
        EvalSample(
            sample_id="bbca_2023_risk",
            company="BBCA",
            year=2023,
            section_type="risk",
            source_text=(
                "Perseroan menghadapi risiko suku bunga dari tren kenaikan Fed Funds Rate "
                "yang berdampak pada biaya dana dan margin bunga bersih. Risiko kredit "
                "dijaga ketat dengan NPL gross 1,9% dan NPL net 0,5%. Risiko likuiditas "
                "dikelola dengan LCR 299% dan NSFR 153%. Risiko operasional meliputi "
                "keamanan siber dan ketergantungan pada sistem teknologi informasi."
            ),
            reference_summary=(
                "Risiko utama BCA 2023 meliputi: risiko suku bunga (Fed Rate), risiko kredit "
                "terkendali (NPL gross 1,9%), risiko likuiditas aman (LCR 299%), dan "
                "risiko operasional terutama siber dan ketergantungan teknologi."
            ),
        ),
        EvalSample(
            sample_id="bbca_2023_outlook",
            company="BBCA",
            year=2023,
            section_type="outlook",
            source_text=(
                "BCA menargetkan pertumbuhan kredit 10-12% pada tahun 2024, didukung oleh "
                "pemulihan ekonomi domestik dan ekspansi ke segmen UMKM digital. "
                "Perseroan akan memperkuat platform myBCA untuk meningkatkan transaksi "
                "digital yang sudah mencapai 99% dari total transaksi. Target ROE "
                "dipertahankan di atas 23% dengan efisiensi biaya operasional (BOPO) "
                "di bawah 40%."
            ),
            reference_summary=(
                "BCA menargetkan pertumbuhan kredit 10-12% di 2024 dengan fokus UMKM digital. "
                "Strategi utama: penguatan platform myBCA (99% transaksi digital), "
                "ROE >23%, dan BOPO <40%."
            ),
        ),
        EvalSample(
            sample_id="bbri_2023_financial",
            company="BBRI",
            year=2023,
            section_type="financial",
            source_text=(
                "BRI membukukan laba bersih Rp 60,4 triliun tahun 2023, tumbuh 17,5% YoY. "
                "Total kredit BRI tumbuh 12,2% menjadi Rp 1.266,6 triliun, dengan segmen "
                "mikro sebagai tulang punggung tumbuh 15,8% ke Rp 626 triliun. "
                "NIM BRI tercatat 8,18%, tertinggi di antara bank BUKU IV. "
                "Kualitas aset membaik dengan NPL gross turun ke 2,84%."
            ),
            reference_summary=(
                "BRI membukukan laba bersih Rp 60,4 T (+17,5% YoY), kredit Rp 1.266,6 T "
                "(+12,2%) dengan segmen mikro Rp 626 T (+15,8%). NIM 8,18% tertinggi "
                "di BUKU IV. NPL gross membaik ke 2,84%."
            ),
        ),
        EvalSample(
            sample_id="bbri_2023_risk",
            company="BBRI",
            year=2023,
            section_type="risk",
            source_text=(
                "Risiko kredit BRI terutama berasal dari segmen mikro dan UKM yang lebih "
                "rentan terhadap perubahan ekonomi makro. Perseroan membentuk pencadangan "
                "CKPN yang memadai dengan coverage ratio 218%. Risiko pasar dikelola "
                "dengan portofolio obligasi senilai Rp 200 triliun yang terekspos terhadap "
                "perubahan yield. Risiko operasional BRI mencakup fraud dan kejahatan "
                "keuangan di jaringan AgenBRILink yang tersebar di 67.000 desa."
            ),
            reference_summary=(
                "Risiko utama BRI: kredit mikro/UKM rentan ekonomi makro (CKPN coverage 218%), "
                "market risk dari portofolio obligasi Rp 200 T, dan risiko operasional "
                "fraud di 67.000 agen AgenBRILink."
            ),
        ),
        EvalSample(
            sample_id="bmri_2023_financial",
            company="BMRI",
            year=2023,
            section_type="financial",
            source_text=(
                "Bank Mandiri membukukan laba bersih Rp 55,1 triliun tahun 2023, "
                "meningkat 33,6% dari Rp 41,2 triliun tahun sebelumnya. "
                "Total aset mencapai Rp 2.174 triliun, menempatkan Mandiri sebagai "
                "bank terbesar di Indonesia berdasarkan aset. Kredit tumbuh 15,5% "
                "menjadi Rp 1.392 triliun. Biaya kredit (CoC) turun ke 0,89% "
                "dari 1,35% pada 2022."
            ),
            reference_summary=(
                "Bank Mandiri mencetak laba bersih Rp 55,1 T (+33,6% YoY), "
                "menjadi bank terbesar Indonesia dengan total aset Rp 2.174 T. "
                "Kredit tumbuh 15,5% ke Rp 1.392 T dengan cost of credit membaik "
                "ke 0,89%."
            ),
        ),
        EvalSample(
            sample_id="tlkm_2023_financial",
            company="TLKM",
            year=2023,
            section_type="financial",
            source_text=(
                "Telkom Indonesia membukukan pendapatan Rp 149,2 triliun tahun 2023, "
                "tumbuh 2,1% YoY. Laba bersih Rp 24,5 triliun, turun 7,6% akibat "
                "kenaikan biaya operasional dan depresiasi. Segmen IndiHome berkontribusi "
                "Rp 28,5 triliun atau 19% dari total pendapatan. EBITDA margin "
                "terjaga di 50,3%."
            ),
            reference_summary=(
                "Telkom 2023: pendapatan Rp 149,2 T (+2,1%) namun laba bersih turun 7,6% "
                "ke Rp 24,5 T akibat biaya naik. IndiHome berkontribusi 19% (Rp 28,5 T). "
                "EBITDA margin stabil di 50,3%."
            ),
        ),
        EvalSample(
            sample_id="asii_2023_financial",
            company="ASII",
            year=2023,
            section_type="financial",
            source_text=(
                "Astra International membukukan laba bersih Rp 33,6 triliun pada 2023, "
                "turun 5% dari tahun sebelumnya Rp 35,4 triliun akibat normalisasi "
                "harga komoditas. Segmen otomotif tetap dominan dengan penjualan "
                "Toyota dan Daihatsu mencapai 651.000 unit. Segmen jasa keuangan "
                "tumbuh 12% dengan total pembiayaan Rp 175 triliun."
            ),
            reference_summary=(
                "Astra 2023: laba bersih Rp 33,6 T (-5% YoY) akibat normalisasi komoditas. "
                "Otomotif tetap dominan (651.000 unit Toyota/Daihatsu). Jasa keuangan "
                "tumbuh 12% dengan pembiayaan Rp 175 T."
            ),
        ),
        EvalSample(
            sample_id="bbni_2023_financial",
            company="BBNI",
            year=2023,
            section_type="financial",
            source_text=(
                "BNI membukukan laba bersih Rp 20,9 triliun tahun 2023, tumbuh 14,2% YoY. "
                "Kredit tumbuh 7,8% menjadi Rp 695 triliun, dengan kredit korporasi "
                "mendominasi 52% portofolio. Fee based income tumbuh 16% menjadi "
                "Rp 9,2 triliun. Rasio NPL gross membaik ke 2,0% dari 2,8% tahun lalu."
            ),
            reference_summary=(
                "BNI 2023: laba bersih Rp 20,9 T (+14,2%), kredit Rp 695 T (+7,8%) "
                "didominasi korporasi 52%. Fee based income +16% ke Rp 9,2 T. "
                "Kualitas aset membaik, NPL gross turun ke 2,0%."
            ),
        ),
        EvalSample(
            sample_id="goto_2023_financial",
            company="GOTO",
            year=2023,
            section_type="financial",
            source_text=(
                "GoTo Group membukukan pendapatan bersih Rp 8,3 triliun tahun 2023, "
                "tumbuh 32% YoY. Rugi bersih menyempit signifikan ke Rp 8,0 triliun "
                "dari Rp 40,5 triliun pada 2022 seiring program efisiensi. GTV "
                "(Gross Transaction Value) mencapai Rp 731 triliun. Segmen financial "
                "technology GoPay menjadi kontributor pertumbuhan terbesar dengan "
                "GTV naik 42%."
            ),
            reference_summary=(
                "GoTo 2023: pendapatan bersih Rp 8,3 T (+32%), rugi bersih menyempit "
                "drastis ke Rp 8,0 T dari Rp 40,5 T berkat efisiensi. GTV Rp 731 T, "
                "GoPay tumbuh 42%."
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SummaryEvaluator:
    """
    Two-layer evaluation system: ROUGE + Claude-as-judge.

    Usage:
        evaluator = SummaryEvaluator(output_dir="data/processed")
        dataset = get_builtin_eval_dataset()
        report = evaluator.evaluate_dataset(dataset)
        print(report.markdown_table())
        evaluator.save_report(report)
    """

    def __init__(
        self,
        judge_model: str = "claude-sonnet-4-20250514",
        output_dir: str | Path = "data/processed",
    ) -> None:
        """
        Args:
            judge_model: Claude model used as judge.
            output_dir:  Directory for saving reports.
        """
        if not HAS_ROUGE:
            raise ImportError(
                "rouge-score required. Run: pip install rouge-score"
            )
        self.judge_model = judge_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        dataset: list[EvalSample],
        skip_judge: bool = False,
    ) -> EvalReport:
        """
        Run full evaluation pipeline on a list of EvalSamples.

        Args:
            dataset:    List of EvalSample objects with source + reference.
            skip_judge: If True, skip Claude-as-judge (ROUGE only, no API cost).

        Returns:
            EvalReport with aggregated metrics and per-sample results.
        """
        logger.info(
            "Starting evaluation | samples={} | judge={}",
            len(dataset), not skip_judge
        )

        results: list[EvalResult] = []

        for i, sample in enumerate(dataset):
            logger.info(
                "Evaluating sample {}/{} | id={}",
                i + 1, len(dataset), sample.sample_id
            )
            result = self._evaluate_sample(sample, skip_judge=skip_judge)
            results.append(result)

        report = self._build_report(results)
        logger.success(
            "Evaluation complete | avg_rouge_L={:.3f} | avg_judge={:.2f}",
            report.avg_rougeL_f1,
            report.avg_overall_judge,
        )
        return report

    def evaluate_single(
        self,
        source_text: str,
        candidate_summary: str,
        reference_summary: str,
        sample_id: str = "custom",
    ) -> EvalResult:
        """
        Evaluate a single candidate summary against a reference.

        Args:
            source_text:        Original source document.
            candidate_summary:  AI-generated summary to evaluate.
            reference_summary:  Ground truth reference summary.
            sample_id:          Label for this evaluation.

        Returns:
            EvalResult with ROUGE and judge scores.
        """
        sample = EvalSample(
            sample_id=sample_id,
            source_text=source_text,
            reference_summary=reference_summary,
        )
        return self._evaluate_sample(
            sample,
            candidate_override=candidate_summary,
            skip_judge=False,
        )

    def save_report(
        self,
        report: EvalReport,
        filename: Optional[str] = None,
    ) -> Path:
        """Save EvalReport as JSON file."""
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_report_{ts}.json"
        path = self.output_dir / filename
        path.write_text(report.to_json(), encoding="utf-8")
        logger.success("Evaluation report saved → {}", path)
        return path

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _evaluate_sample(
        self,
        sample: EvalSample,
        candidate_override: Optional[str] = None,
        skip_judge: bool = False,
    ) -> EvalResult:
        """Run evaluation on a single EvalSample."""
        input_tokens = output_tokens = 0

        # Step 1: Generate candidate summary (or use override)
        if candidate_override:
            candidate = candidate_override
        else:
            gen_result = analyze_financial_text(
                sample.source_text,
                task=FinancialTask.SUMMARIZE,
            )
            candidate = gen_result["result"]
            input_tokens += gen_result["usage"]["input_tokens"]
            output_tokens += gen_result["usage"]["output_tokens"]

        # Step 2: ROUGE scores
        rouge = self._compute_rouge(candidate, sample.reference_summary)

        # Step 3: Claude-as-judge
        if skip_judge:
            judge = JudgeScores()
        else:
            judge_result = self._judge(
                sample.source_text, candidate, sample.reference_summary
            )
            judge = judge_result["scores"]
            input_tokens += judge_result["input_tokens"]
            output_tokens += judge_result["output_tokens"]

        return EvalResult(
            sample_id=sample.sample_id,
            company=sample.company,
            year=sample.year,
            source_chars=len(sample.source_text),
            candidate_summary=candidate,
            reference_summary=sample.reference_summary,
            rouge=rouge,
            judge=judge,
            eval_model=self.judge_model,
            evaluated_at=datetime.now().isoformat(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _compute_rouge(
        self, candidate: str, reference: str
    ) -> RougeScores:
        """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
        scores = self._rouge.score(reference, candidate)
        r1 = scores["rouge1"]
        r2 = scores["rouge2"]
        rl = scores["rougeL"]

        length_ratio = (
            len(candidate) / len(reference)
            if reference else 0.0
        )

        return RougeScores(
            rouge1_precision=round(r1.precision, 4),
            rouge1_recall=round(r1.recall, 4),
            rouge1_f1=round(r1.fmeasure, 4),
            rouge2_precision=round(r2.precision, 4),
            rouge2_recall=round(r2.recall, 4),
            rouge2_f1=round(r2.fmeasure, 4),
            rougeL_precision=round(rl.precision, 4),
            rougeL_recall=round(rl.recall, 4),
            rougeL_f1=round(rl.fmeasure, 4),
            length_ratio=round(length_ratio, 3),
        )

    def _judge(
        self,
        source_text: str,
        candidate: str,
        reference: str,
    ) -> dict:
        """Call Claude-as-judge to grade the candidate summary."""
        user_message = JUDGE_USER_TEMPLATE.format(
            source_text=source_text[:2000],
            candidate_summary=candidate[:1500],
            reference_summary=reference[:800],
        )

        from src.api.client import get_client
        response = get_client().messages.create(
            model=self.judge_model,
            max_tokens=512,
            temperature=0.0,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        raw = response.content[0].text.strip()
        scores = self._parse_judge_response(raw)

        return {
            "scores": scores,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def _parse_judge_response(self, raw: str) -> JudgeScores:
        """Parse JSON response from Claude judge."""
        try:
            # Strip markdown code fences if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            data = json.loads(clean.strip())

            fa = int(data.get("factual_accuracy", 0))
            fr = int(data.get("financial_relevance", 0))
            co = int(data.get("completeness", 0))
            cn = int(data.get("conciseness", 0))

            overall = (
                fa * SCORE_WEIGHTS["factual_accuracy"]
                + fr * SCORE_WEIGHTS["financial_relevance"]
                + co * SCORE_WEIGHTS["completeness"]
                + cn * SCORE_WEIGHTS["conciseness"]
            )

            return JudgeScores(
                factual_accuracy=fa,
                financial_relevance=fr,
                completeness=co,
                conciseness=cn,
                overall=round(overall, 3),
                reasoning=data.get("reasoning", ""),
                raw_response=raw,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Failed to parse judge response: {} | raw={}", exc, raw[:200])
            return JudgeScores(raw_response=raw)

    def _build_report(self, results: list[EvalResult]) -> EvalReport:
        """Aggregate per-sample results into an EvalReport."""
        n = len(results)
        if n == 0:
            return EvalReport(
                report_id="empty",
                created_at=datetime.now().isoformat(),
                n_samples=0,
                eval_model=self.judge_model,
            )

        def avg(vals: list[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return EvalReport(
            report_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            created_at=datetime.now().isoformat(),
            n_samples=n,
            eval_model=self.judge_model,
            avg_rouge1_f1=avg([r.rouge.rouge1_f1 for r in results]),
            avg_rouge2_f1=avg([r.rouge.rouge2_f1 for r in results]),
            avg_rougeL_f1=avg([r.rouge.rougeL_f1 for r in results]),
            avg_factual_accuracy=avg([r.judge.factual_accuracy for r in results]),
            avg_financial_relevance=avg([r.judge.financial_relevance for r in results]),
            avg_completeness=avg([r.judge.completeness for r in results]),
            avg_conciseness=avg([r.judge.conciseness for r in results]),
            avg_overall_judge=avg([r.judge.overall for r in results]),
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            results=results,
        )
