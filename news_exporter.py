#!/usr/bin/env python3
"""
news_exporter.py
Export article+summary pairs from a news dataset (cnn_dailymail or xsum) into (docX.pdf, docX.ref.txt).
"""
import os
import argparse
import pathlib
from datasets import load_dataset  # pip install datasets
from fpdf import FPDF              # pip install fpdf

def save_pdf(text, pdf_path, title=""):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    if title:
        # Replace characters not supported in latin-1
        pdf.multi_cell(0, 8, title.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(4)
    pdf.set_font("Arial", size=12)
    for paragraph in text.split("\n\n"):
        for line in paragraph.split("\n"):
            pdf.multi_cell(0, 6, line.encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(2)
    pdf.output(str(pdf_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_docs_news", help="Output folder")
    ap.add_argument("--dataset", choices=["cnn_dailymail", "xsum"], default="cnn_dailymail")
    ap.add_argument("--split", default="validation", help="Dataset split")
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cnn_dailymail":
        ds = load_dataset("cnn_dailymail", "3.0.0", split=args.split)
        art_field, sum_field, title_field = "article", "highlights", None
    else:
        ds = load_dataset("xsum", split=args.split)
        art_field, sum_field, title_field = "document", "summary", "id"

    ds = ds.shuffle(seed=args.seed).select(range(min(args.count, len(ds))))

    saved = 0
    for i, item in enumerate(ds, 1):
        article = item[art_field].strip()
        summary = item[sum_field].strip()
        title = item.get(title_field, "") or ""
        pdf_path = out / f"news_{i:05d}.pdf"
        ref_path = out / f"news_{i:05d}.ref.txt"
        save_pdf(article, pdf_path, title=title)
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(summary)
        saved += 1
        if saved % 25 == 0:
            print(f"Saved {saved}...")
    print(f"Done. Saved {saved} pairs to {out}")

if __name__ == "__main__":
    main()
