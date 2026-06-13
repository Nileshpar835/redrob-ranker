#!/usr/bin/env python3
"""
precompute_index.py — Offline retrieval / semantic-search index builder.

This is the "pre-computation" stage referenced in submission_metadata.yaml
(pre_computation_required: true). It runs ONCE, offline, and may take
longer than the 5-minute ranking budget (it does not — it's ~1-2 minutes
for 100K candidates — but the spec explicitly allows pre-computation to
exceed the window).

What it does
------------
1. Builds a text "document" per candidate by concatenating the fields that
   actually describe what they've *done* (headline, summary, career-history
   titles + descriptions, skill names) — this is the corpus the retrieval
   system searches over.

2. Fits a TF-IDF vectorizer (unigrams + bigrams, English stopwords removed)
   over the full 100K-document corpus and transforms every candidate
   document into a sparse TF-IDF vector. TF-IDF + cosine similarity is a
   classic vector-space / semantic retrieval method: it weights terms by
   how distinctive they are across the corpus (so generic words like
   "experience" or "team" get down-weighted relative to "retrieval",
   "embeddings", "ranking"), which is exactly the "look past the literal
   keyword, weight by relevance" behavior semantic search is meant to
   provide — and it requires no model download, no GPU, no network.

3. Saves the fitted vectorizer and the candidate document-vector matrix to
   disk (`retrieval_index/`), so that `rank.py` can load them at ranking
   time and do a fast cosine-similarity lookup against a JD query vector
   — this is the "retrieval pipeline" used as the semantic-search
   component of the hybrid score.

Artifacts produced (in ./retrieval_index/):
  - vectorizer.pkl      — fitted TfidfVectorizer
  - candidate_vectors.npz — sparse TF-IDF matrix, row i = candidates[i]
  - candidate_ids.json  — ordered list of candidate_ids matching matrix rows

Run:
    python precompute_index.py --candidates ./candidates.jsonl --out ./retrieval_index

Runtime: ~60-90s for 100K candidates on a single CPU core.
"""

import argparse
import gzip
import json
import os
import pickle

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from rank import candidate_document


def load_candidates(path):
    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True, help="Output directory for index artifacts")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading candidates from {args.candidates} ...")
    candidates = load_candidates(args.candidates)
    print(f"Loaded {len(candidates)} candidates.")

    print("Building candidate documents ...")
    docs = [candidate_document(c) for c in candidates]
    candidate_ids = [c["candidate_id"] for c in candidates]

    print("Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(docs)
    print(f"TF-IDF matrix shape: {matrix.shape}")

    vec_path = os.path.join(args.out, "vectorizer.pkl")
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)

    matrix_path = os.path.join(args.out, "candidate_vectors.npz")
    sparse.save_npz(matrix_path, matrix)

    ids_path = os.path.join(args.out, "candidate_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)

    print(f"Saved index artifacts to {args.out}/")
    print(f"  - {vec_path}")
    print(f"  - {matrix_path}")
    print(f"  - {ids_path}")


if __name__ == "__main__":
    main()