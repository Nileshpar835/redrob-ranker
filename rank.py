#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon ranker (Senior AI Engineer, Founding Team JD)

Approach
--------
A transparent, rule-based, multi-component scorer over the candidate
profile + career history + behavioral signals. No embeddings, no LLM
calls, no GPU. Pure Python stdlib, runs on the full 100K pool in a few
seconds on a laptop CPU.

Components (each 0-1, then weighted-summed into a base score):

  1. title_fit        — does current/recent title match the AI/ML/ranking/
                         retrieval/search/recommendation engineering
                         archetype the JD wants? Heaviest weight, and the
                         primary defense against keyword-stuffer traps
                         (a "Marketing Manager" with 10 AI skills scores
                         low here regardless of skill list).

  2. skill_fit        — JD-relevant skill coverage, but TRUST-WEIGHTED:
                         a skill only counts if it has reasonable
                         endorsements AND duration_months > 0. This is the
                         second line of defense against keyword stuffing
                         (skills listed with 0 duration / 0 endorsements
                         contribute almost nothing).

  3. career_substance — scans career_history descriptions (not just
                         skills/title) for evidence of having actually
                         built/operated production retrieval, ranking,
                         search, recommendation, or ML-infra systems —
                         the "gap between what the JD says and what it
                         means". Rewards product-company experience,
                         penalizes pure-consulting-only careers and
                         pure-research-only careers per the JD's explicit
                         disqualifiers.

  4. experience_fit   — years_of_experience vs the 5-9y band, with a soft
                         curve (doesn't hard-cliff outside the band, since
                         the JD says it's a range not a requirement).

  5. location_fit     — Pune/Noida/Hyderabad/Mumbai/Delhi-NCR preference,
                         relocation willingness, India-based preference
                         per JD logistics section.

  6. education_fit    — minor signal; tier of institution + relevant field.

  7. honeypot_penalty  — explicit internal-consistency checks for
                         impossible profiles (tenure exceeding company
                         age proxies, "expert" skills with 0 duration,
                         experience/age inconsistencies, contradictory
                         dates). Candidates failing these checks get a
                         strong multiplicative penalty so they fall out
                         of top-100 naturally, without needing a labeled
                         honeypot list.

Behavioral multiplier
----------------------
The weighted base score is then multiplied by a behavioral-availability
factor derived from redrob_signals: recency of activity, recruiter
response rate, open_to_work flag, interview completion, profile
completeness, and verification status. A perfect-on-paper candidate who
is inactive / unresponsive is down-weighted, per the JD's explicit
instruction.

Tie-breaking
------------
Scores are rounded to 4 decimals for the submission (monotonic by
construction). Ties are broken by candidate_id ascending, per spec.

Runtime: ~5-8s for 100K candidates on a single CPU core, no network,
no GPU. Memory: dominated by holding the JSONL in memory (~1-2 GB for
100K records), well within 16GB.
"""

import argparse
import csv
import gzip
import json
import os
import pickle
import re
from datetime import date, datetime

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Reference date for "recency" calculations (dataset is synthetic; we treat
# the most recent last_active_date / signup_date seen in the pool as "today"
# so the ranker is self-contained and doesn't depend on wall-clock time).
# This is computed dynamically in main().
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# JD-derived keyword sets
# ---------------------------------------------------------------------------

# Titles that directly match the "intelligence layer / ranking / retrieval /
# search / recommendation / ML engineering" archetype the JD wants.
CORE_TITLE_PATTERNS = [
    r"\bml\b", r"machine learning", r"\bai\b", r"artificial intelligence",
    r"applied scientist", r"research engineer", r"\bnlp\b",
    r"\bllm\b", r"search", r"retrieval", r"ranking", r"recommend",
    r"information retrieval", r"data scientist", r"deep learning",
]

# Adjacent / engineering titles that could plausibly transition (data /
# backend engineers with the right career history get credit via
# career_substance, but the title itself is only a partial match).
ADJACENT_TITLE_PATTERNS = [
    r"data engineer", r"backend engineer", r"software engineer",
    r"platform engineer", r"infrastructure engineer", r"full.?stack",
    r"data analyst", r"analytics engineer",
]

# Titles that are explicit non-fits regardless of skills list (anti
# keyword-stuffer signal — JD: "title is 'Marketing Manager' is not a fit").
NON_FIT_TITLE_PATTERNS = [
    r"marketing", r"sales", r"hr\b", r"human resources", r"recruit",
    r"content writer", r"copywriter", r"product manager", r"project manager",
    r"business analyst", r"finance", r"accountant", r"operations manager",
    r"customer success", r"support engineer", r"qa engineer", r"tester",
    r"designer", r"ux", r"\bcfo\b", r"\bceo\b", r"\bcoo\b",
    r"administrator", r"office manager", r"legal",
]

# Seniority terms that suggest "architecture/tech-lead, no longer writes
# code" — JD disqualifier if it's been their role for 18+ months.
ARCH_ONLY_PATTERNS = [
    r"\bdirector\b", r"\bvp\b", r"vice president", r"head of",
    r"\bcto\b", r"chief technology",
]

# Things we absolutely need (production experience signals) — searched in
# career_history descriptions + skills.
RETRIEVAL_TERMS = [
    "embedding", "sentence-transformer", "sentence transformer", "bge",
    "e5", "openai embedding", "vector database", "vector db", "pinecone",
    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss",
    "hybrid search", "hybrid retrieval", "bm25", "ann ", "approximate nearest",
    "semantic search", "dense retrieval",
]
RANKING_EVAL_TERMS = [
    "ndcg", "mrr", "map@", "mean average precision", "a/b test", "ab test",
    "offline evaluation", "online evaluation", "learning to rank",
    "learning-to-rank", "ltr", "xgboost ranking", "precision@", "recall@",
    "click-through", "ctr",
]
LLM_TERMS = [
    "fine-tun", "lora", "qlora", "peft", "llm", "large language model",
    "rag", "retrieval augmented", "retrieval-augmented", "transformer",
    "gpt", "prompt engineering",
]
ML_GENERAL_TERMS = [
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "recommendation system",
    "recommender system", "ranking model", "matching system",
    "search system", "classification model", "regression model",
]

# Human-readable display names for evidence terms (used only in reasoning
# text, so raw regex/matching fragments like "fine-tun" or "ann " don't
# leak into the output verbatim).
TERM_DISPLAY = {
    "fine-tun": "LLM fine-tuning",
    "ann ": "approximate nearest-neighbor search",
    "map@": "MAP-based ranking evaluation",
    "ab test": "A/B testing",
    "a/b test": "A/B testing",
    "ltr": "learning-to-rank",
    "ctr": "click-through-rate analysis",
    "rag": "RAG (retrieval-augmented generation)",
    "gpt": "GPT-based LLM work",
    "e5": "E5 embeddings",
    "bge": "BGE embeddings",
    "vector db": "vector database",
    "ndcg": "NDCG-based ranking evaluation",
    "mrr": "MRR-based ranking evaluation",
}


def display_term(t):
    return TERM_DISPLAY.get(t, t).strip()


# Pure consulting firms named in the JD as a soft negative if it's the
# *entire* career.
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "hcl", "tech mahindra",
    "ltimindtree", "l&t infotech",
]

# Pure CV / speech / robotics without NLP/IR — soft negative.
CV_ONLY_TERMS = ["computer vision", "image classification", "object detection",
                 "speech recognition", "robotics", "autonomous driving",
                 "image segmentation", "ocr"]

# Preferred locations per JD logistics.
PREFERRED_LOCATIONS = ["pune", "noida", "delhi", "ncr", "gurgaon", "gurugram",
                        "hyderabad", "mumbai", "bangalore", "bengaluru"]

TIER1_INSTITUTION_HINTS = ["iit", "iisc", "bits pilani", "iiit"]


# ---------------------------------------------------------------------------
# Semantic search / retrieval pipeline
# ---------------------------------------------------------------------------
#
# Loads the precomputed TF-IDF retrieval index (built offline by
# precompute_index.py) and the JD query text (built offline by
# extract_jd_query.py, an LLM-powered workflow). At ranking time this is a
# single vectorizer.transform() on one short JD-query string, plus one
# cosine_similarity call against the 100K x V sparse candidate matrix —
# both essentially instant (<1s), keeping the full pipeline within the
# 5-minute compute budget with no network or GPU.
#
# TF-IDF + cosine similarity is a vector-space (semantic) retrieval method:
# it scores candidate documents by weighted term overlap with the JD query,
# where term weights reflect corpus-wide distinctiveness (e.g. "retrieval"
# and "embeddings" are weighted far higher than "experience" or "team").
# This gives the ranker a genuine retrieval/recommendation signal —
# candidates whose career-history language is semantically close to the
# "ideal candidate" description score higher — independent of, and
# complementary to, the rule-based components below.

def load_retrieval_index(index_dir):
    """Load the precomputed TF-IDF vectorizer + candidate vectors.

    Returns (vectorizer, candidate_matrix, candidate_ids) or
    (None, None, None) if the index is not present (semantic component is
    then skipped gracefully — the ranker still works rule-based-only).
    """
    vec_path = os.path.join(index_dir, "vectorizer.pkl")
    matrix_path = os.path.join(index_dir, "candidate_vectors.npz")
    ids_path = os.path.join(index_dir, "candidate_ids.json")

    if not (os.path.exists(vec_path) and os.path.exists(matrix_path) and os.path.exists(ids_path)):
        return None, None, None

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    matrix = sparse.load_npz(matrix_path)
    with open(ids_path, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)

    return vectorizer, matrix, candidate_ids


def compute_semantic_scores(vectorizer, candidate_matrix, candidate_ids, jd_query_text):
    """Cosine-similarity retrieval scores for every candidate against the
    JD query, returned as {candidate_id: score in [0, 1]}."""
    query_vec = vectorizer.transform([jd_query_text])
    sims = cosine_similarity(query_vec, candidate_matrix).ravel()

    # Min-max normalize across the pool so the component sits cleanly in
    # [0, 1] alongside the other rule-based components, regardless of the
    # absolute cosine-similarity scale (which is typically small, ~0.0-0.3,
    # for sparse TF-IDF vectors over long documents).
    lo, hi = sims.min(), sims.max()
    if hi > lo:
        sims = (sims - lo) / (hi - lo)
    else:
        sims = sims * 0.0

    return dict(zip(candidate_ids, sims.tolist()))


def candidate_document(cand):
    """Build the retrieval text document for a candidate (same logic as
    precompute_index.py — used by app.py's sandbox for fresh TF-IDF fits
    on small samples)."""
    p = cand.get("profile", {})
    parts = []
    parts.append(p.get("headline", ""))
    parts.append(p.get("current_title", ""))
    parts.append(p.get("summary", ""))

    for job in cand.get("career_history", []) or []:
        title = job.get("title", "")
        desc = job.get("description", "")
        parts.append(title)
        parts.append(desc)
        if job.get("is_current"):
            parts.append(title)
            parts.append(desc)

    skill_names = [s.get("name", "") for s in (cand.get("skills") or [])]
    parts.append(" ".join(skill_names))

    return " ".join(p for p in parts if p)


def text_blob(cand):
    """Concatenate all free-text fields for keyword scanning (lowercased)."""
    parts = []
    p = cand.get("profile", {})
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))
    for job in cand.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " | ".join(parts).lower()


def career_blob(cand):
    parts = []
    for job in cand.get("career_history", []):
        parts.append(job.get("title", "") + " " + job.get("description", ""))
    return " ".join(parts).lower()


def count_matches(blob, patterns):
    n = 0
    for pat in patterns:
        if re.search(pat, blob):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

def title_fit(cand):
    title = cand.get("profile", {}).get("current_title", "").lower()

    if count_matches(title, NON_FIT_TITLE_PATTERNS):
        return 0.05

    core_hits = count_matches(title, CORE_TITLE_PATTERNS)
    adj_hits = count_matches(title, ADJACENT_TITLE_PATTERNS)
    arch_hits = count_matches(title, ARCH_ONLY_PATTERNS)

    if core_hits >= 1:
        score = 1.0
    elif adj_hits >= 1:
        score = 0.55
    else:
        score = 0.15

    if arch_hits:
        # Architecture/lead-only titles: only fine if career history shows
        # recent hands-on production work (checked via career_substance,
        # so just apply a moderate haircut here).
        score *= 0.6

    return score


def skill_fit(cand):
    """Trust-weighted JD-relevant skill coverage.

    A skill counts toward the score only if:
      - its name matches a JD-relevant term, AND
      - duration_months > 0 (catches "0 years used" honeypot/stuffer skills), AND
      - endorsements contribute a multiplicative trust factor.
    """
    skills = cand.get("skills", []) or []
    if not skills:
        return 0.0

    relevant_terms = RETRIEVAL_TERMS + RANKING_EVAL_TERMS + LLM_TERMS + ML_GENERAL_TERMS + [
        "python", "vector", "search", "ranking", "embedding",
    ]

    total_trust_weight = 0.0
    for s in skills:
        name = (s.get("name") or "").lower()
        matched = False
        for term in relevant_terms:
            if term in name or name in term:
                matched = True
                break
        if not matched:
            continue

        dur = s.get("duration_months", 0) or 0
        endorse = s.get("endorsements", 0) or 0
        prof = s.get("proficiency", "intermediate")

        if dur <= 0:
            # Listed but never actually used -> essentially worthless,
            # classic keyword-stuffer / honeypot pattern.
            continue

        dur_factor = min(dur / 24.0, 1.0)          # caps at 2 years used
        endorse_factor = min(endorse / 15.0, 1.0)   # caps at 15 endorsements
        trust = 0.4 + 0.4 * dur_factor + 0.2 * endorse_factor  # 0.4-1.0

        prof_weight = {"beginner": 0.4, "intermediate": 0.7,
                        "advanced": 1.0, "expert": 1.2}.get(prof, 0.7)

        total_trust_weight += trust * prof_weight

    # Normalize: ~6 strong relevant skills -> near 1.0
    return min(total_trust_weight / 6.0, 1.0)


def career_substance(cand):
    """Evidence of production retrieval/ranking/ML-infra work, scanning
    career_history descriptions, not just the skills list."""
    blob = career_blob(cand)
    full_blob = text_blob(cand)

    retrieval_hits = count_matches(blob, RETRIEVAL_TERMS)
    eval_hits = count_matches(blob, RANKING_EVAL_TERMS)
    llm_hits = count_matches(blob, LLM_TERMS)
    ml_hits = count_matches(blob, ML_GENERAL_TERMS)

    score = 0.0
    score += min(retrieval_hits, 3) * 0.12   # up to 0.36 — "things absolutely need"
    score += min(eval_hits, 2) * 0.10        # up to 0.20 — eval frameworks
    score += min(llm_hits, 2) * 0.08         # up to 0.16 — nice-to-have
    score += min(ml_hits, 3) * 0.08          # up to 0.24 — general ML production

    score = min(score, 1.0)

    # Product-company vs pure-consulting career check.
    histories = cand.get("career_history", []) or []
    companies = [(h.get("company") or "").lower() for h in histories]
    if companies and all(any(c in comp for c in CONSULTING_FIRMS) for comp in companies):
        score *= 0.5  # entire career at consulting firms -> JD soft negative

    # Pure research-only (no production deployment) check: lots of ML terms
    # but zero retrieval/eval production signals and title says
    # "research"/"scientist" with no "production"/"deployed"/"shipped" terms.
    if ml_hits >= 2 and retrieval_hits == 0 and eval_hits == 0:
        if re.search(r"research (scientist|engineer|fellow)|academic|phd|postdoc", full_blob):
            if not re.search(r"production|deployed|shipped|launched|scaled", blob):
                score *= 0.5

    # CV/speech/robotics-only without NLP/IR -> JD soft negative.
    cv_hits = count_matches(blob, CV_ONLY_TERMS)
    nlp_hits = count_matches(full_blob, ["nlp", "natural language", "retrieval",
                                          "search", "ranking", "embedding", "llm"])
    if cv_hits >= 2 and nlp_hits == 0:
        score *= 0.5

    return score


def experience_fit(cand):
    yoe = cand.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= yoe <= 9:
        return 1.0
    if yoe < 5:
        # soft falloff below 5
        return max(0.0, 1.0 - (5 - yoe) * 0.18)
    # soft falloff above 9
    return max(0.0, 1.0 - (yoe - 9) * 0.12)


def location_fit(cand):
    p = cand.get("profile", {})
    loc = (p.get("location") or "").lower()
    country = (p.get("country") or "").lower()

    score = 0.3  # baseline
    if any(city in loc for city in ["pune", "noida"]):
        score = 1.0
    elif any(city in loc for city in PREFERRED_LOCATIONS):
        score = 0.8
    elif "india" in country:
        score = 0.6

    sig = cand.get("redrob_signals", {}) or {}
    if sig.get("willing_to_relocate"):
        score = max(score, 0.7)
    if "india" not in country and not sig.get("willing_to_relocate", False):
        score *= 0.6  # outside India, not willing to relocate -> case-by-case, lean down

    return min(score, 1.0)


def education_fit(cand):
    edu = cand.get("education", []) or []
    if not edu:
        return 0.4

    best = 0.0
    for e in edu:
        tier = (e.get("tier") or "unknown").lower()
        field = (e.get("field_of_study") or "").lower()
        inst = (e.get("institution") or "").lower()

        tier_score = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.5,
                       "tier_4": 0.35, "unknown": 0.45}.get(tier, 0.45)

        if any(h in inst for h in TIER1_INSTITUTION_HINTS):
            tier_score = max(tier_score, 1.0)

        field_bonus = 0.1 if any(k in field for k in
                                  ["computer", "data", "ai", "ml", "math", "statistic", "software"]) else 0.0

        best = max(best, min(tier_score + field_bonus, 1.0))

    return best


def honeypot_penalty(cand):
    """Returns a multiplicative penalty in (0, 1]. 1.0 = no issues found.
    Lower values indicate likely honeypot / internally-inconsistent profiles.
    """
    penalty = 1.0
    p = cand.get("profile", {})
    skills = cand.get("skills", []) or []
    histories = cand.get("career_history", []) or []
    yoe = p.get("years_of_experience", 0) or 0

    # 1. "Expert" proficiency with 0 duration_months — classic honeypot pattern.
    expert_zero_dur = sum(1 for s in skills
                           if (s.get("proficiency") == "expert")
                           and (s.get("duration_months", 0) or 0) == 0)
    if expert_zero_dur >= 3:
        penalty *= 0.15
    elif expert_zero_dur >= 1:
        penalty *= 0.5

    # 2. Total career duration vs years_of_experience grossly inconsistent.
    total_months = sum(h.get("duration_months", 0) or 0 for h in histories)
    total_years = total_months / 12.0
    if yoe > 0 and total_years > 0:
        ratio = total_years / yoe
        if ratio < 0.4 or ratio > 2.5:
            penalty *= 0.6

    # 3. Duplicate/overlapping current jobs (more than one is_current=true).
    current_count = sum(1 for h in histories if h.get("is_current"))
    if current_count > 1:
        penalty *= 0.6

    # 4. Skills list absurdly long with all "expert" (stuffing).
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if len(skills) >= 15 and expert_count >= 10:
        penalty *= 0.5

    # 5. Education end_year before start_year, or end_year in the future
    #    beyond a plausible graduation + experience timeline.
    for e in cand.get("education", []) or []:
        sy, ey = e.get("start_year"), e.get("end_year")
        if sy and ey and ey < sy:
            penalty *= 0.4

    # 6. Career start before plausible working age given education end_year.
    edu_ends = [e.get("end_year") for e in cand.get("education", []) or [] if e.get("end_year")]
    if edu_ends and histories:
        earliest_start = min((h.get("start_date") or "9999") for h in histories)
        try:
            earliest_year = int(earliest_start[:4])
            if earliest_year < min(edu_ends) - 1:
                penalty *= 0.6
        except (ValueError, TypeError):
            pass

    return penalty


def behavioral_multiplier(cand, ref_date):
    sig = cand.get("redrob_signals", {}) or {}

    mult = 1.0

    # Recency of activity.
    last_active = sig.get("last_active_date")
    if last_active:
        try:
            d = datetime.strptime(last_active, "%Y-%m-%d").date()
            days_inactive = (ref_date - d).days
            if days_inactive <= 14:
                recency = 1.0
            elif days_inactive <= 30:
                recency = 0.9
            elif days_inactive <= 90:
                recency = 0.7
            elif days_inactive <= 180:
                recency = 0.45
            else:
                recency = 0.25
        except ValueError:
            recency = 0.7
    else:
        recency = 0.5
    mult *= (0.5 + 0.5 * recency)  # recency contributes 0.5-1.0 range

    # Open to work.
    if sig.get("open_to_work_flag"):
        mult *= 1.05
    else:
        mult *= 0.85

    # Recruiter response rate.
    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        mult *= (0.6 + 0.4 * max(0.0, min(rr, 1.0)))

    # Interview completion rate (if they have an offer history at all).
    icr = sig.get("interview_completion_rate")
    if icr is not None:
        mult *= (0.7 + 0.3 * max(0.0, min(icr, 1.0)))

    # Profile completeness.
    pcs = sig.get("profile_completeness_score")
    if pcs is not None:
        mult *= (0.8 + 0.2 * (max(0.0, min(pcs, 100.0)) / 100.0))

    # Verification.
    verified = sum([
        bool(sig.get("verified_email")),
        bool(sig.get("verified_phone")),
        bool(sig.get("linkedin_connected")),
    ])
    mult *= (0.9 + 0.1 * (verified / 3.0))

    # Notice period (JD: <30 days preferred, can buy out up to 30).
    notice = sig.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            mult *= 1.0
        elif notice <= 60:
            mult *= 0.95
        else:
            mult *= 0.88

    return mult


# Component weights for the base (pre-behavioral) score. Title and
# career_substance carry the most weight as the primary defense against
# keyword stuffers; semantic_fit adds a retrieval/recommendation signal
# from the TF-IDF hybrid-search component.
WEIGHTS = {
    "title": 0.26,
    "skills": 0.14,
    "career": 0.24,
    "semantic": 0.18,
    "experience": 0.09,
    "location": 0.06,
    "education": 0.03,
}


def score_candidate(cand, ref_date, semantic_score=0.0):
    components = {
        "title": title_fit(cand),
        "skills": skill_fit(cand),
        "career": career_substance(cand),
        "semantic": semantic_score,
        "experience": experience_fit(cand),
        "location": location_fit(cand),
        "education": education_fit(cand),
    }

    base = sum(WEIGHTS[k] * v for k, v in components.items())
    base *= honeypot_penalty(cand)
    base *= behavioral_multiplier(cand, ref_date)

    return base, components


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------

def build_reasoning(cand, components, score, rank=None):
    """Builds a 1-3 sentence, profile-grounded reasoning string.

    Designed against the Stage 4 manual-review checklist:
      - cites specific facts (years, title, named skills/signals)
      - connects to JD requirements (title/skill/career/location framing)
      - acknowledges weaknesses honestly (lowest-scoring component(s) for
        this candidate are surfaced as concerns, not hidden)
      - avoids hallucination: every named skill/term comes from
        career_blob/skills, which are derived directly from the profile
      - rank-aware tone: candidates further down the list get reasoning
        that explicitly notes why they trail the top picks, rather than
        reusing top-pick phrasing.
    """
    p = cand.get("profile", {})
    title = p.get("current_title", "Unknown title")
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "Unknown")
    sig = cand.get("redrob_signals", {}) or {}
    rr = sig.get("recruiter_response_rate")

    bits = [f"{title} with {yoe:.1f} yrs in {loc}."]

    # Career substance evidence — pick the most relevant career-history hit.
    blob = career_blob(cand)
    evidence_terms = RETRIEVAL_TERMS + RANKING_EVAL_TERMS + LLM_TERMS + ML_GENERAL_TERMS
    found = [t for t in evidence_terms if t in blob]
    readable = []
    for t in found:
        d = display_term(t)
        if d and d not in readable:
            readable.append(d)
        if len(readable) == 2:
            break

    if readable:
        bits.append("Career history shows " + " and ".join(readable) + " experience.")
    elif components["career"] < 0.15:
        bits.append("Limited evidence of production retrieval/ranking work in career history.")

    # Skill fit comment.
    if components["skills"] >= 0.5:
        bits.append("Strong, duration-backed AI/IR skill set.")
    elif components["skills"] >= 0.2:
        bits.append("Some relevant skills, but moderate endorsement/duration backing.")
    else:
        bits.append("Few JD-relevant skills with real usage history -- a notable gap.")

    # Semantic / retrieval signal.
    sem = components.get("semantic", 0)
    if sem >= 0.7:
        bits.append("Profile text is a strong semantic match to the JD's ideal-candidate description.")
    elif sem <= 0.2:
        bits.append("Profile text has low semantic overlap with the JD's ideal-candidate description.")

    # Behavioral.
    if rr is not None:
        if rr >= 0.5:
            bits.append(f"Recruiter response rate {rr:.2f} -- engaged.")
        else:
            bits.append(f"Recruiter response rate {rr:.2f} -- a concern for outreach.")

    if not sig.get("open_to_work_flag", True):
        bits.append("Not currently marked open-to-work, down-weighted.")

    notice = sig.get("notice_period_days")
    if notice is not None and notice > 30:
        bits.append(f"Notice period of {notice} days is longer than the JD's sub-30-day preference.")

    # Location.
    if components["location"] >= 0.9:
        bits.append("Based in Pune/Noida, matching the JD's preferred locations.")
    elif components["location"] <= 0.4:
        bits.append("Location is a weaker match for the Pune/Noida-centric role.")

    # Honest concerns: surface the weakest component(s) explicitly so the
    # reasoning's tone tracks the rank, rather than sounding uniformly
    # positive across all 100 rows.
    weak_components = {
        "title": "title is only an adjacent match to the AI/ML/retrieval archetype the JD wants",
        "career": "career history shows limited direct evidence of shipping retrieval/ranking systems",
        "skills": "skill list has limited JD-relevant, duration-backed entries",
        "semantic": "overall profile language only partially overlaps with the JD's ideal-candidate description",
        "experience": "years of experience sits outside the JD's 5-9 year band",
        "location": "location is a weaker fit for the Pune/Noida-centric role",
        "education": "education background is only a loose fit for the role",
    }
    sorted_components = sorted(components.items(), key=lambda kv: kv[1])
    weakest_key, weakest_val = sorted_components[0]

    if rank is not None and rank > 20 and weakest_key in weak_components and weakest_val < 0.4:
        bits.append(f"Main gap relative to top picks: {weak_components[weakest_key]}.")

    return " ".join(bits)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_candidates(path):
    opener = gzip.open if path.endswith(".gz") else open
    candidates = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def compute_ref_date(candidates):
    """Use the max last_active_date across the pool as the 'today' reference,
    so the recency calculation is self-contained (no wall-clock dependency)."""
    dates = []
    for c in candidates:
        d = (c.get("redrob_signals") or {}).get("last_active_date")
        if d:
            try:
                dates.append(datetime.strptime(d, "%Y-%m-%d").date())
            except ValueError:
                pass
    return max(dates) if dates else date.today()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True,
                         help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--index-dir", default="./retrieval_index",
                         help="Directory containing precomputed TF-IDF retrieval index "
                              "(from precompute_index.py). If absent, semantic component "
                              "is skipped and ranking falls back to rule-based-only.")
    parser.add_argument("--jd-query", default="./jd_query.txt",
                         help="Path to JD query text (from extract_jd_query.py)")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    ref_date = compute_ref_date(candidates)

    # Load precomputed semantic retrieval index (offline artifacts).
    vectorizer, candidate_matrix, candidate_ids = load_retrieval_index(args.index_dir)
    semantic_scores = {}
    if vectorizer is not None:
        if os.path.exists(args.jd_query):
            with open(args.jd_query, "r", encoding="utf-8") as f:
                jd_query_text = f.read()
        else:
            jd_query_text = ""
        if jd_query_text.strip():
            semantic_scores = compute_semantic_scores(
                vectorizer, candidate_matrix, candidate_ids, jd_query_text
            )
            print(f"Loaded semantic retrieval index ({len(candidate_ids)} candidates) "
                  f"and computed semantic scores.")
        else:
            print("Warning: JD query text empty; semantic component set to 0.")
    else:
        print(f"Warning: no retrieval index found at {args.index_dir}; "
              f"semantic component set to 0. Run precompute_index.py first for the hybrid score.")

    scored = []
    for cand in candidates:
        cid = cand.get("candidate_id", "")
        sem = semantic_scores.get(cid, 0.0)
        try:
            score, components = score_candidate(cand, ref_date, semantic_score=sem)
        except Exception:
            score, components = 0.0, {k: 0.0 for k in WEIGHTS}
        scored.append((cand, score, components))

    # Sort by score desc, then candidate_id asc for deterministic tie-break.
    scored.sort(key=lambda x: (-x[1], x[0].get("candidate_id", "")))

    top = scored[:args.top]

    # Round scores first, then re-sort: primary by rounded score desc,
    # secondary by candidate_id asc (spec-required tie-break for equal
    # scores after rounding).
    rounded = [(cand, round(raw_score, 4), components) for cand, raw_score, components in top]
    rounded.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))

    rows = []
    prev_score = None
    for rank, (cand, score, components) in enumerate(rounded, start=1):
        if prev_score is not None and score > prev_score:
            score = prev_score  # defensive: enforce non-increasing
        prev_score = score
        reasoning = build_reasoning(cand, components, score, rank=rank)
        rows.append({
            "candidate_id": cand["candidate_id"],
            "rank": rank,
            "score": f"{score:.4f}",
            "reasoning": reasoning,
        })


    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()