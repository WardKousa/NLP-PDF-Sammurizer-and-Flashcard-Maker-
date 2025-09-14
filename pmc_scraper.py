#!/usr/bin/env python3
"""
pmc_scraper.py
Download PDFs + abstracts from PubMed Central (PMC Open Access subset) as (docX.pdf, docX.ref.txt) pairs.

Usage (examples):
  python pmc_scraper.py --out eval_docs_pmc --count 200 --term "machine learning" --email you@example.com
  python pmc_scraper.py --out eval_docs_pmc --count 200 --term "natural language processing" --email you@example.com

Notes:
- Targets PMC (Open Access) so that PDFs are downloadable.
- Uses NCBI E-utilities. Provide a contact email (NCBI policy).
- Requires: pip install requests
"""
import os
import re
import time
import argparse
import pathlib
import requests
import xml.etree.ElementTree as ET

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_ARTICLE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
PMC_PDF = "https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"

def safe_name(s, max_len=80):
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[^A-Za-z0-9 _.-]', '', s)
    return s[:max_len] or "untitled"

def esearch_pmc(term, retmax, email):
    params = {
        "db": "pmc",
        "term": f"{term} AND hasabstract[text]",
        "retmax": str(retmax),
        "retmode": "xml",
        "email": email
    }
    r = requests.get(ESEARCH, params=params, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ids = [e.text for e in root.findall(".//IdList/Id")]
    return ids

def fetch_abstract_and_title(pmcid, email):
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
        "email": email
    }
    r = requests.get(EFETCH, params=params, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    # Extract title
    title_el = root.find(".//article-title")
    title = "".join(title_el.itertext()).strip() if title_el is not None else f"PMC{pmcid}"
    # Extract abstract
    abs_el = root.find(".//abstract")
    abstract = ""
    if abs_el is not None:
        abstract = " ".join(" ".join(p.itertext()).strip() for p in abs_el.findall(".//p"))
        abstract = re.sub(r'\s+', ' ', abstract).strip()
    return title, abstract

def download_pdf(pmcid, path):
    url = PMC_PDF.format(pmcid=pmcid)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<15):
                if chunk:
                    f.write(chunk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_docs_pmc", help="Output folder")
    ap.add_argument("--count", type=int, default=200, help="How many articles to fetch")
    ap.add_argument("--term", default="machine learning", help="Search term")
    ap.add_argument("--email", required=True, help="Contact email for NCBI E-utilities")
    ap.add_argument("--per_sec", type=float, default=1.0, help="Max requests per second")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[PMC] Searching for '{args.term}'...")
    ids = esearch_pmc(args.term, args.count * 2, args.email)  # fetch extra to account for failures
    print(f"[PMC] Found {len(ids)} candidate IDs")

    saved = 0
    for idx, id_ in enumerate(ids, 1):
        pmcid = id_
        if not pmcid.startswith("PMC"):
            pmcid = "PMC" + pmcid  # normalize

        try:
            title, abstract = fetch_abstract_and_title(pmcid, args.email)
            if not abstract:
                continue
            base = safe_name(f"{pmcid}_{title}")
            pdf_path = out / f"{base}.pdf"
            ref_path = out / f"{base}.ref.txt"
            if pdf_path.exists() and ref_path.exists():
                print(f"skip exists: {base}")
                continue
            # Save abstract
            with open(ref_path, "w", encoding="utf-8") as f:
                f.write(abstract)
            # Download PDF
            download_pdf(pmcid, pdf_path)
            saved += 1
            print(f"[{saved}] saved {pdf_path.name}")
            time.sleep(1.0 / max(args.per_sec, 0.2))
            if saved >= args.count:
                break
        except Exception as e:
            print("error:", pmcid, e)
            # cleanup partials
            try:
                if pdf_path.exists() and pdf_path.stat().st_size < 10_000:
                    pdf_path.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"Done. Saved {saved} pairs to {out}")

if __name__ == "__main__":
    main()
