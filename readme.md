# Redrob Hackathon — Hybrid Candidate Ranker

Hybrid (rule-based + semantic retrieval) ranker for the *Senior AI
Engineer — Founding Team* JD, scored against 100,000 candidates. Produces
a top-100 ranking CSV matching `submission_spec.md`.

## Architecture

Two stages, matching the brief's "AI ranking systems, semantic search,
retrieval pipelines, recommendation systems, LLM-powered workflows":

```
                  OFFLINE (one-time, can exceed 5 min)
+-----------------------+   +-----------------------------------+
| extract_jd_query.py    |   | precompute_index.py                |
| (LLM-powered workflow) |   | (retrieval pipeline / embeddings)  |
| JD text -> distilled   |   | 100K candidates -> TF-IDF vectors  |
| "ideal candidate"      |   | -> retrieval_index/                |
| query -> jd_query.txt  |   |                                     |
+-----------+------------+   +------------------+------------------+
            |                                    |
            +-----------------+------------------+
                              |
                              v
                  ONLINE (ranking run, <5 min, CPU, no network)
                  +------------------------------------+
                  | rank.py                              |
                  |  - load retrieval_index +            |
                  |    jd_query.txt                      |
                  |  - cosine similarity (semantic       |
                  |    search component, ~1s)            |
                  |  - hybrid score = semantic +          |
                  |    rule-based components              |
                  |    (title/career/skills/honeypot/     |
                  |     behavioral)                       |
                  |  -> submission.csv                    |
                  +------------------------------------+
```

## Quick start

```bash
pip install -r requirements.txt

# Offline, one-time (or whenever the JD / candidate pool changes):
python extract_jd_query.py                    # writes jd_query.txt
python precompute_index.py --candidates ./candidates.jsonl --out ./retrieval_index

# Ranking run (the reproduce_command):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

- **Ranking runtime**: ~40s for 100K candidates on a single CPU core
  (semantic similarity lookup is <1s; the rest is the existing rule-based
  scan).
- **Offline pre-computation**: ~60-90s to build the TF-IDF index (declared
  in `submission_metadata.yaml` as `pre_computation_required: true`).
- No network calls, no GPU, no per-candidate LLM calls at ranking time.
- `rank.py` degrades gracefully to rule-based-only if `retrieval_index/`
  is missing (prints a warning, semantic component = 0).

## Components

| Component | Weight | What it checks |
|---|---|---|
| `title_fit` | 0.26 | Does the current title match the AI/ML/ranking/retrieval archetype? Explicit non-fit titles (Marketing Manager, HR, etc.) are heavily down-weighted regardless of skills -- primary defense against keyword stuffers. |
| `career_substance` | 0.24 | Scans **career history descriptions** for evidence of production retrieval, ranking, evaluation, LLM, or ML-infra work. Penalizes pure-consulting-only and pure-research-only careers per the JD's explicit disqualifiers. |
| `semantic` | 0.18 | **TF-IDF cosine-similarity retrieval score** between the candidate's full profile/career text and an LLM-distilled "ideal candidate" query derived from the JD's "How to read between the lines" section. This is the hybrid-search / recommendation-system component -- it catches candidates whose language is semantically aligned with the role even if exact keywords differ. |
| `skill_fit` | 0.14 | JD-relevant skills, **trust-weighted** by `duration_months` and `endorsements` -- a skill listed with 0 months used contributes ~nothing. |
| `experience_fit` | 0.09 | Soft curve centered on the JD's 5-9 year band. |
| `location_fit` | 0.06 | Pune/Noida preferred, other Tier-1 Indian cities partial credit. |
| `education_fit` | 0.03 | Institution tier + relevance of field of study. |

Then two multiplicative adjustments (unchanged from v1):

- **`honeypot_penalty`** (0.15-1.0): internal-consistency checks (expert
  skills with 0 duration, career/experience-year mismatches, overlapping
  current jobs, education timeline inconsistencies).
- **`behavioral_multiplier`** (~0.2-1.3): activity recency, recruiter
  response rate, open-to-work flag, interview completion, profile
  completeness, verification status.

## Why TF-IDF instead of neural embeddings

The compute environment for this task has no network access (so
`sentence-transformers` / HuggingFace model downloads aren't possible) and
the ranking step itself must run with no network and no GPU. TF-IDF +
cosine similarity is a genuine vector-space / semantic retrieval method --
the same family as dense embeddings, just with sparse, corpus-statistic
weighted vectors instead of a neural encoder -- and it requires zero model
downloads, fits comfortably offline, and the lookup at ranking time is
near-instant.

If a `sentence-transformers` model were available locally (e.g. bundled
into the repo / Docker image), `precompute_index.py` could be swapped to
produce dense embeddings instead, and `rank.py`'s `compute_semantic_scores`
would be unchanged (it's already just "vectorize query, cosine similarity
against precomputed matrix") -- the architecture is designed so that swap
is a one-file change.

## The "LLM-powered workflow"

`extract_jd_query.py` represents an LLM-powered step in the pipeline: an
LLM (Claude) read the ~1500-word JD -- including its meta-commentary about
hackathon traps, its "things we explicitly do NOT want" section, and its
"how to read between the lines" ideal-candidate description -- and
distilled it into a dense, retrieval-focused query (`jd_query.txt`). This
reasons about *what the JD means* (the gap the JD itself calls out)
before that meaning becomes the retrieval query. It runs once, offline;
no LLM calls happen during the 100K-candidate ranking run.

## Tie-breaking

Scores are rounded to 4 decimals, then candidates are re-sorted by
`(score desc, candidate_id asc)` so any ties created by rounding satisfy
the spec's tie-break rule. `validate_submission.py` confirms this.

## Files

- `rank.py` -- hybrid ranker (reproduce_command target)
- `precompute_index.py` -- offline TF-IDF retrieval index builder
- `extract_jd_query.py` -- offline LLM-derived JD query
- `retrieval_index/` -- precomputed artifacts (vectorizer, candidate
  vectors, candidate IDs). **Not included in this file bundle** (the
  candidate-vectors matrix is ~120MB for 100K candidates) -- regenerate
  with `python precompute_index.py --candidates ./candidates.jsonl --out ./retrieval_index`
  (~60-90s) and commit to your repo for the `reproduce_command` to work
  as a single step.
- `jd_query.txt` -- the distilled JD query used for semantic scoring
- `app.py` -- Gradio sandbox (HF Spaces); fits a fresh small TF-IDF index
  on the uploaded sample for a self-contained demo
- `submission.csv` -- generated top-100 ranking, validated
- `requirements.txt`
- `submission_metadata.yaml`

## Known limitations / future work

- TF-IDF captures lexical/statistical semantic overlap, not deep semantic
  equivalence (e.g. "built a job-matching engine" vs "built a
  recommendation system" share fewer n-grams than a neural embedding would
  recognize). A local sentence-transformer model would likely improve
  recall on paraphrased descriptions -- the architecture supports this
  swap without other changes.
- `behavioral_multiplier` and `honeypot_penalty` thresholds remain
  heuristic (unchanged from v1); a labeled validation sample would allow
  calibration.
- The semantic component currently treats the JD query as fixed; a
  retrieval-augmented re-ranking pass (LLM re-ranks top-N by semantic
  score) was considered but excluded to keep ranking-time LLM calls at
  zero, per compute constraints.