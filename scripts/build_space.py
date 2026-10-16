"""Assemble a clean Hugging Face Space from this repo and (optionally) upload it.

    python scripts/build_space.py                          # build/space + checks, no upload
    python scripts/build_space.py --push                   # upload, replacing every old file
    python scripts/build_space.py --push --set-config      # also set Space variables
    python scripts/build_space.py --push --set-api-key     # also store ANTHROPIC_API_KEY as a secret

The old Space was uploaded by hand: it carried __pycache__, .DS_Store and an
index built by the buggy chunker. This script uploads exactly the files the
app needs, refuses to build without a valid index manifest, and deletes
anything else in the Space.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.indexer import DEFAULT_EMBED_MODEL  # noqa: E402
from src.rag.manifest import check_compatible  # noqa: E402

DEFAULT_SPACE = "Agatha1911/finsight-idx"
GRADIO_VERSION = "6.12.0"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

SPACE_README = """---
title: FinSight IDX
emoji: 📊
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: {gradio}
app_file: app.py
pinned: false
license: mit
short_description: RAG Q&A over IDX annual reports with page citations
---

# FinSight IDX

Ask questions about an Indonesian annual report and get answers with page
citations. Built with hybrid retrieval (multilingual embeddings + BM25) and Claude.

{status}

Source, evaluation and limitations: https://github.com/Agathahah/finsight-idx

Not investment advice. Check every figure against the cited page.
"""

VARIABLES = {
    "FINSIGHT_STRICT_INDEX": "1",
    "RATE_LIMIT_PER_MINUTE": "5",
    "RATE_LIMIT_GLOBAL_PER_MINUTE": "60",
    "MAX_QUESTION_CHARS": "500",
}


def build(out: Path, persist_dir: Path) -> dict:
    manifest = check_compatible(persist_dir, DEFAULT_EMBED_MODEL, strict=True)
    if not (persist_dir / "chroma.sqlite3").exists():
        sys.exit(f"{persist_dir}/chroma.sqlite3 not found: index the PDF first")

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(ROOT / "app.py", out / "app.py")
    shutil.copytree(ROOT / "src", out / "src", ignore=IGNORE)
    shutil.copytree(persist_dir, out / "data" / "vectorstore", ignore=IGNORE)
    shutil.copy2(ROOT / "requirements_spaces.txt", out / "requirements.txt")

    docs = manifest["documents"].values()
    status = "Indexed: " + "; ".join(
        f"{d['company']} {d['year']} ({d['total_chunks']:,} chunks, {d['total_pages']} pages, "
        f"OCR on {len(d.get('pages_ocr', []))} pages)" for d in docs)
    (out / "README.md").write_text(SPACE_README.format(gradio=GRADIO_VERSION, status=status), encoding="utf-8")
    return manifest


def summarize(out: Path) -> None:
    files = [p for p in out.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"Built {out}: {len(files)} files, {size / 1e6:.1f} MB")
    for p in sorted(files):
        if p.stat().st_size > 1e6:
            print(f"  {p.relative_to(out)}  {p.stat().st_size / 1e6:.1f} MB")
    bad = [p for p in files if "__pycache__" in p.parts or p.name == ".DS_Store" or p.name == ".env"]
    if bad:
        sys.exit(f"Refusing to continue, unwanted files: {bad}")


def push(out: Path, space: str, set_config: bool, set_api_key: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Logged in as {api.whoami()['name']}; uploading to spaces/{space} ...")
    commit = api.upload_folder(
        folder_path=str(out), repo_id=space, repo_type="space",
        commit_message="Deploy: rebuilt index (schema v2) and production serving path",
        delete_patterns=["*"],   # remove files that are no longer part of the app
    )
    print("Uploaded:", commit)
    if set_config:
        variables = {**VARIABLES, "CLAUDE_MODEL": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")}
        for k, v in variables.items():
            api.add_space_variable(space, k, v)
        print("Variables set:", ", ".join(f"{k}={v}" for k, v in variables.items()))
    if set_api_key:
        key = os.getenv("ANTHROPIC_API_KEY", "")
        if not key.startswith("sk-ant-"):
            sys.exit("ANTHROPIC_API_KEY not found in the environment (.env)")
        api.add_space_secret(space, "ANTHROPIC_API_KEY", key)
        print("Secret ANTHROPIC_API_KEY set (value not printed)")
    print(f"Space rebuilds now: https://huggingface.co/spaces/{space}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--space", default=DEFAULT_SPACE)
    ap.add_argument("--out", default=str(ROOT / "build" / "space"))
    ap.add_argument("--persist-dir", default=str(ROOT / "data" / "vectorstore"))
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--set-config", action="store_true")
    ap.add_argument("--set-api-key", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    out = Path(args.out)
    build(out, Path(args.persist_dir))
    summarize(out)
    if args.push:
        push(out, args.space, args.set_config, args.set_api_key)
    else:
        print("Dry run: nothing uploaded. Add --push to deploy.")


if __name__ == "__main__":
    main()
