#!/usr/bin/env python3
"""
arxiv_scraper.py
Download PDFs + abstracts from arXiv as (docX.pdf, docX.ref.txt) pairs.

Usage (examples):
  python arxiv_scraper.py --out eval_docs_arxiv --count 500 --categories cs.CL,cs.LG
  python arxiv_scraper.py --out eval_docs_arxiv --query "large language models" --count 200

Notes:
- Requires: pip install arxiv requests
- arXiv abstracts serve as human-written summaries for benchmarking (standard in many papers).
- Be polite: the script rate-limits requests.
"""
import os
import re
import time
import argparse
import pathlib
import arxiv  # pip install arxiv
import requests

def safe_name(s, max_len=80):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[^A-Za-z0-9 _.-]', '', s)
    return s[:max_len] or "untitled"

def download_pdf(url, path, timeout=60):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<15):
                if chunk:
                    f.write(chunk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_docs_arxiv", help="Output folder")
    ap.add_argument("--count", type=int, default=100, help="How many papers to fetch")
    ap.add_argument("--categories", default="cs.CL", help="Comma-separated arXiv categories (e.g., cs.CL,cs.LG)")
    ap.add_argument("--query", default=None, help="Optional search query to refine results")
    ap.add_argument("--per_sec", type=float, default=1.0, help="Max downloads per second (politeness)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    query_parts = []
    if cats:
        # arxiv library supports category search via query string
        cat_query = " OR ".join(f"cat:{c}" for c in cats)
        query_parts.append(f"({cat_query})")
    if args.query:
        query_parts.append(f"({args.query})")
    query = " AND ".join(query_parts) if query_parts else None

    print(f"[arXiv] Query: {query or '(all)'}")
    print(f"[arXiv] Target count: {args.count}")

    search = arxiv.Search(
        query=query,
        max_results=args.count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    n = 0
    for result in search.results():
        title = result.title or "untitled"
        abstract = (result.summary or "").strip()
        if not abstract:
            continue
        # Build paths
        base_name = safe_name(f"arxiv_{result.entry_id.split('/')[-1]}_{title}")
        pdf_path = out / f"{base_name}.pdf"
        ref_path = out / f"{base_name}.ref.txt"
        if pdf_path.exists() and ref_path.exists():
            print(f"skip exists: {base_name}")
            continue
        try:
            # Write summary
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write(abstract.strip())
            # Download PDF
            pdf_url = result.pdf_url
            download_pdf(pdf_url, pdf_path)
            n += 1
            print(f"[{n}/{args.count}] saved: {pdf_path.name}")
            time.sleep(1.0 / max(args.per_sec, 0.1))
        except Exception as e:
            print("error:", e)
            # Remove partials
            try:
                if pdf_path.exists() and pdf_path.stat().st_size < 10_000:
                    pdf_path.unlink(missing_ok=True)
            except Exception:
                pass
        if n >= args.count:
            break

    print(f"Done. Saved {n} pairs to {out}")

if __name__ == "__main__":
    main()
