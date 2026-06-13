#!/usr/bin/env python3
"""
extract_jd_query.py — LLM-powered workflow (offline, run once).

This script represents the "LLM-powered workflow" piece of the hybrid
system. It is run OFFLINE, during development, NOT at ranking time (the
ranking step itself makes no LLM/network calls, per the compute
constraints).

What it does
------------
Takes the raw job_description text and produces a structured
"ideal candidate" query document — the text that the retrieval pipeline
(precompute_index.py's TF-IDF space) will be embedded into and compared
against every candidate document via cosine similarity.

Why this matters: the JD is ~1500 words of narrative, including sections
about what NOT to want, culture philosophy, and meta-commentary about the
hackathon itself. Naively embedding the *entire* JD text would dilute the
retrieval signal with irrelevant narrative tokens. An LLM reads the JD and
extracts the dense, retrieval-relevant query: the technical/role substance
of "How to read between the lines" plus the "Things you absolutely need"
section plus the production-system framing — i.e. it reasons about *what
the JD means*, not just what it says, before that meaning is fed into the
vector-space retrieval step.

This is a ONE-TIME, offline step. The output (jd_query.txt) is a static
artifact checked into the repo and consumed by rank.py at ranking time —
no LLM calls happen during the 100K-candidate ranking run itself.

In this submission, the extraction below was performed by Claude reading
job_description.md and distilling it into a retrieval query. Declared in
submission_metadata.yaml's ai_usage_summary.
"""

JD_QUERY = """
Senior AI engineer with hands-on production experience building and
operating embeddings-based retrieval systems (sentence-transformers,
OpenAI embeddings, BGE, E5) deployed to real users, including handling
embedding drift, index refresh, and retrieval-quality regression.
Production experience with vector databases or hybrid search
infrastructure: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch,
Elasticsearch, FAISS, BM25 hybrid retrieval. Strong Python, high code
quality. Hands-on experience designing evaluation frameworks for ranking
systems: NDCG, MRR, MAP, offline-to-online correlation, A/B testing.
Has shipped at least one end-to-end ranking, search, recommendation, or
matching system to real users at meaningful scale, at a product company
(not a pure consulting or services firm). Strong opinions on retrieval
architecture (hybrid versus dense), evaluation methodology (offline versus
online), and LLM integration (fine-tuning versus prompting), grounded in
systems actually built and operated. Comfortable with LLM fine-tuning
(LoRA, QLoRA, PEFT), learning-to-rank models (XGBoost or neural),
distributed systems, large-scale inference optimization. Background in
recruiting technology, HR-tech, or marketplace products is a plus.
Title reflects engineering ownership of ranking, retrieval, matching, or
recommendation systems -- machine learning engineer, applied scientist,
search engineer, recommendation systems engineer, ML platform engineer.
Six to nine years total experience, several years in applied ML or AI
roles at product companies. Has written production code recently, not
purely in architecture or tech-lead roles for the last 18 months.
NLP, information retrieval, or search/ranking background, not purely
computer vision, speech, or robotics. Open-source contributions, technical
blog posts, or papers demonstrating systems thinking rather than framework
tutorials. Based in or willing to relocate to Pune, Noida, Delhi NCR,
Hyderabad, or Mumbai, India.
"""


def main():
    with open("jd_query.txt", "w", encoding="utf-8") as f:
        f.write(JD_QUERY.strip() + "\n")
    print("Wrote jd_query.txt")


if __name__ == "__main__":
    main()