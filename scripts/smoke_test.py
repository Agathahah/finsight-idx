"""Smoke test a running FinSight app: locally before deploy, or the live Space after.

    python scripts/smoke_test.py http://127.0.0.1:7860            # free checks only
    python scripts/smoke_test.py http://127.0.0.1:7860 --paid     # + one real question (~$0.01)
    python scripts/smoke_test.py Agatha1911/finsight-idx --paid   # the deployed Space
    python scripts/smoke_test.py URL --paid --expect-rate-limit   # app started with RATE_LIMIT_PER_MINUTE=1

Free checks send inputs the guardrails must reject before any API call.
--paid sends one real question and checks the answer has a figure and a page
citation. --expect-rate-limit sends a second valid question right after and
expects it to be refused (no API cost when the limit is 1 per minute).
Exit code 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

API = "/ask_laporan"
QUESTION = "Berapa laba bersih BCA tahun 2023 yang dapat diatribusikan kepada pemilik entitas induk?"
# 48.639.122 juta = Rp48,6 triliun: accept any of the ways the report writes it
EXPECTED = re.compile(r"48[.,]6|48\.639")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="URL (http://127.0.0.1:7860) or Space id (owner/name)")
    ap.add_argument("--paid", action="store_true", help="send one real question (~$0.01)")
    ap.add_argument("--expect-rate-limit", action="store_true")
    args = ap.parse_args()

    from gradio_client import Client
    client = Client(args.target, verbose=False)
    results: list[tuple[str, bool, str]] = []

    def check(name: str, inputs: tuple, ok, show: int = 90) -> str:
        t0 = time.perf_counter()
        out = client.predict(*inputs, api_name=API)
        passed = bool(ok(out))
        results.append((name, passed, f"{time.perf_counter() - t0:4.1f}s  {out[:show]!r}"))
        return out

    check("too short is refused", ("hi", "", ""), lambda o: o.startswith("⚠️") and "pendek" in o)
    check("too long is refused", ("laba " * 120, "", ""), lambda o: o.startswith("⚠️") and "panjang" in o)
    check("bad ticker is refused", ("Berapa laba bersih?", "BCA1", ""), lambda o: "4 huruf" in o)
    check("bad year is refused", ("Berapa laba bersih?", "BBCA", "1990"), lambda o: "rentang" in o)

    if args.paid:
        out = check("real question: figure + page citation", (QUESTION, "BBCA", "2023"),
                    lambda o: "## Jawaban" in o and "hal." in o and EXPECTED.search(o) and "id `" in o,
                    show=160)
        if "Kuota harian" in out:
            print("note: the daily budget is used up; the paid check cannot pass until it resets")
    if args.expect_rate_limit:
        check("second question within a minute is rate limited", ("Berapa total aset BCA 2023?", "BBCA", "2023"),
              lambda o: o.startswith("⏳"))

    width = max(len(n) for n, _, _ in results)
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, p, _ in results if not p]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
