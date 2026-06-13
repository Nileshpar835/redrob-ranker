"""
app.py — Sandbox demo for HuggingFace Spaces (Gradio).

Accepts a small JSONL sample of candidates (<=100), runs rank.py's hybrid
scoring logic end-to-end (rule-based components + TF-IDF semantic
retrieval against the precomputed JD query), and returns a ranked CSV.

For the sandbox, the semantic component is computed against the small
uploaded sample directly (no precomputed index needed) — a fresh TF-IDF
fit on the sample + JD query, so the demo is self-contained.
"""

import csv
import io
import json

import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank import score_candidate, build_reasoning, compute_ref_date, candidate_document


def load_jd_query():
    try:
        with open("jd_query.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def rank_sample(file_obj):
    if file_obj is None:
        return "Please upload a .jsonl file of candidate profiles.", None

    candidates = []
    with open(file_obj.name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    if not candidates:
        return "No candidates found in file.", None

    ref_date = compute_ref_date(candidates)

    # Compute semantic scores via a fresh TF-IDF fit on this small sample
    # (mirrors precompute_index.py / rank.py's hybrid logic, but
    # self-contained for a sandbox demo of <=100 docs).
    jd_query = load_jd_query()
    semantic_scores = {}
    if jd_query:
        docs = [candidate_document(c) for c in candidates]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
        matrix = vectorizer.fit_transform(docs)
        query_vec = vectorizer.transform([jd_query])
        sims = cosine_similarity(query_vec, matrix).ravel()
        lo, hi = sims.min(), sims.max()
        if hi > lo:
            sims = (sims - lo) / (hi - lo)
        else:
            sims = sims * 0.0
        semantic_scores = {c["candidate_id"]: float(s) for c, s in zip(candidates, sims)}

    scored = []
    for cand in candidates:
        sem = semantic_scores.get(cand.get("candidate_id", ""), 0.0)
        try:
            score, components = score_candidate(cand, ref_date, semantic_score=sem)
        except Exception:
            score, components = 0.0, {}
        scored.append((cand, score, components))

    scored.sort(key=lambda x: (-round(x[1], 4), x[0].get("candidate_id", "")))

    top_n = min(100, len(scored))
    top = scored[:top_n]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()

    prev_score = None
    for rank, (cand, raw_score, components) in enumerate(top, start=1):
        score = round(raw_score, 4)
        if prev_score is not None and score > prev_score:
            score = prev_score
        prev_score = score
        reasoning = build_reasoning(cand, components, score, rank=rank)
        writer.writerow({
            "candidate_id": cand["candidate_id"],
            "rank": rank,
            "score": f"{score:.4f}",
            "reasoning": reasoning,
        })

    csv_text = buf.getvalue()

    out_path = "ranked_sample.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(csv_text)

    preview = "\n".join(csv_text.splitlines()[:11])
    return preview, out_path


demo = gr.Interface(
    fn=rank_sample,
    inputs=gr.File(label="Upload candidates sample (.jsonl)", file_types=[".jsonl", ".json"]),
    outputs=[
        gr.Textbox(label="Preview (first 10 rows)", lines=12),
        gr.File(label="Download ranked CSV"),
    ],
    title="Redrob Hackathon — Hybrid Ranker Sandbox",
    description=(
        "Upload a small sample of candidate JSONL records (<=100) to see the "
        "hybrid ranker score and rank them end-to-end: rule-based "
        "title/career/skill/honeypot/behavioral components plus a TF-IDF "
        "semantic-retrieval component scored against the JD's "
        "ideal-candidate description. CPU-only, no network, no GPU, "
        "completes in seconds."
    ),
)

if __name__ == "__main__":
    demo.launch()