# summarizer.py
import re
import time
from typing import List, Tuple, Optional
import PyPDF2
from transformers import pipeline, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import sys
import os

# Default summarization model (small-ish; change if you want)
SUM_MODEL = "sshleifer/distilbart-cnn-12-6"

# Detect device: if CUDA available use first GPU (device index 0), else CPU (-1)
DEVICE = 0 if torch.cuda.is_available() else -1

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF. Returns empty string on error."""
    texts = []
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                try:
                    t = p.extract_text()
                except Exception:
                    t = ""
                if t:
                    texts.append(t)
    except Exception as e:
        print(f"[extract_text_from_pdf] ERROR reading {path}: {e}", file=sys.stderr)
        return ""
    return "\n".join(texts)

def split_to_sentences(text: str) -> List[str]:
    """Naive sentence splitter (works OK for English documents)."""
    sents = re.split(r'(?<=[\.!\?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 0]

def chunk_sentences(sentences: List[str], max_chars: int = 1000, overlap_sentences: int = 2) -> List[Tuple[int, List[str]]]:
    """Chunk list of sentences into (start_index, [sentences]) chunks."""
    chunks = []
    cur = []
    cur_len = 0
    idx = 0
    while idx < len(sentences):
        s = sentences[idx]
        if cur_len + len(s) <= max_chars or len(cur) == 0:
            cur.append(s)
            cur_len += len(s)
            idx += 1
        else:
            chunks.append((idx - len(cur), cur.copy()))
            idx = max(0, idx - overlap_sentences)
            cur = []
            cur_len = 0
    if cur:
        chunks.append((idx - len(cur), cur.copy()))
    return chunks

def join_sentences(sent_list: List[str]) -> str:
    return " ".join(sent_list)

# cache the pipeline so it's loaded only once
_SUMMARIZER_CACHE: Optional[Pipeline] = None

def get_summarizer(model_name: str = SUM_MODEL, device: int = DEVICE, verbose: bool = True) -> Pipeline:
    """Return a transformers pipeline for summarization (cached)."""
    global _SUMMARIZER_CACHE
    if _SUMMARIZER_CACHE is not None:
        return _SUMMARIZER_CACHE

    if verbose:
        dev = f"cuda:{device}" if device >= 0 else "cpu"
        print(f"[summarizer] Loading summarizer model '{model_name}' on device: {dev}")

    try:
        _SUMMARIZER_CACHE = pipeline("summarization", model=model_name, device=device)
    except Exception as e:
        # If model fails to load with device=int (GPU), try CPU fallback so script doesn't die.
        print(f"[summarizer] Warning: failed to load pipeline on device {device}: {e}", file=sys.stderr)
        if device >= 0:
            print("[summarizer] Retrying on CPU...", file=sys.stderr)
            _SUMMARIZER_CACHE = pipeline("summarization", model=model_name, device=-1)
        else:
            raise
    if verbose:
        print("[summarizer] Model loaded!")
    return _SUMMARIZER_CACHE

def summarize_text(text: str, summarizer: Pipeline, max_chars: int = 1000, batch_size: int = 4) -> str:
    """
    Chunk the document into pieces and summarize each chunk in batches.
    Returns final stitched summary (and optionally a short second-pass condense).
    """
    if not text:
        return ""

    sentences = split_to_sentences(text)
    chunks = chunk_sentences(sentences, max_chars=max_chars, overlap_sentences=2)
    chunk_texts = [join_sentences(c[1]) for c in chunks]

    summaries: List[str] = []

    # Process in batches for GPU efficiency
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        try:
            out = summarizer(batch, truncation=True, min_length=30, max_length=150)
        except TypeError:
            # If pipeline doesn't accept list (older transformers), run sequentially
            out = [summarizer(b, truncation=True, min_length=30, max_length=150)[0] for b in batch]
        # normalise output to list of summary_text
        summaries.extend([o['summary_text'] if isinstance(o, dict) and 'summary_text' in o else (o[0]['summary_text'] if isinstance(o, list) else str(o)) for o in out])

    final = " ".join(summaries).strip()
    # optional short second-pass to condense if very long
    if len(final) > 600:
        try:
            out = summarizer(final, truncation=True, min_length=30, max_length=200)
            final = out[0]['summary_text']
        except Exception:
            pass
    return final

def align_citations(original_text: str, summary: str, top_k: int = 1):
    """
    Map each summary sentence to the most similar source sentences using TF-IDF + cosine similarity.
    Returns list of (summary_sentence, [(source_sentence, source_index, sim_score), ...])
    """
    source_sents = split_to_sentences(original_text)
    summary_sents = split_to_sentences(summary)
    if not source_sents or not summary_sents:
        return []

    vec = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        src_mat = vec.fit_transform(source_sents)
    except Exception:
        return []

    results = []
    for s in summary_sents:
        qv = vec.transform([s])
        sims = cosine_similarity(qv, src_mat)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        entries = [(source_sents[i], i, float(sims[i])) for i in top_idx]
        results.append((s, entries))
    return results

def generate_flashcards(original_text: str, n_cards: int = 6):
    sents = split_to_sentences(original_text)
    if not sents:
        return []
    vec = TfidfVectorizer(stop_words='english', max_features=200)
    X = vec.fit_transform(sents)
    keywords = vec.get_feature_names_out()
    avg_scores = np.asarray(X.mean(axis=0)).ravel()
    ranked = sorted(zip(keywords, avg_scores), key=lambda x: x[1], reverse=True)
    chosen = [k for k, _ in ranked[:min(n_cards, len(ranked))]]
    cards = []
    for kw in chosen:
        ans = next((s for s in sents if kw.lower() in s.lower()), "")
        q = f"What is {kw}?"
        a = ans if ans else "See summary for context."
        cards.append({"q": q, "a": a})
    return cards

def summarize_with_latency(pdf_path: str, summarizer: Optional[Pipeline] = None, model_name: str = SUM_MODEL, batch_size: int = 4):
    """
    Convenience wrapper that extracts text, optionally creates a summarizer, and summarizes while timing.
    Returns (summary, latency_seconds, extracted_text)
    """
    txt = extract_text_from_pdf(pdf_path)
    if summarizer is None:
        summarizer = get_summarizer(model_name=model_name)
    t0 = time.perf_counter()
    out = summarize_text(txt, summarizer, batch_size=batch_size)
    latency = time.perf_counter() - t0
    return out, latency, txt
