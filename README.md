# Simple Recommendation Engine

## Overview

This project demonstrates the fundamentals of a recommendation engine, a core technology used by platforms like Netflix, Amazon, and Spotify to suggest items to users. The engine uses **collaborative filtering** to find users with similar preferences and recommends items they've highly rated. It's designed as an educational tool to understand how real recommendation systems work, with simple, math-forward code and no external dependencies beyond Python's standard library.

## How It Works

The recommendation engine runs through a 4-stage pipeline:

1. **Similarity Calculator** — Compute how similar two users are based on their rating patterns
2. **Candidate Generator** — Find similar users and collect items they've highly rated but the target user hasn't
3. **Scorer** — Score each candidate based on a similarity-weighted average of how neighbors rated it
4. **Evaluator** — Measure recommendation quality using Precision@K metric

```
User A's Ratings
      ↓
[Similarity Calculator] ← User B, C, D ratings
      ↓
Find similar users (User B, C)
      ↓
[Candidate Generator]
      ↓
Collect high-rated items from similar users that User A hasn't rated
      ↓
[Scorer]
      ↓
Compute weighted scores for each candidate item
      ↓
[Evaluator]
      ↓
Measure how many recommended items are truly relevant
      ↓
Final Ranked Recommendations
```

## Components

### similarity.py
**What it does:** Computes similarity measures between users (cosine similarity) and items (Jaccard similarity).

**Key methods:**
- `cosine_similarity(vec1: dict, vec2: dict) -> float` — Measures rating vector similarity
- `jaccard_similarity(set1: set, set2: set) -> float` — Measures tag set overlap

**Edge cases handled:** Zero vectors (returns 0.0), empty sets (returns 1.0 if both empty), missing keys (treated as 0).

### candidate_generator.py
**What it does:** Finds similar users and generates candidate items they've rated highly but the target user hasn't tried yet. This is the core of collaborative filtering.

**Key methods:**
- `generate_candidates(target_user, user_ratings, item_tags, top_n_neighbors=2) -> set` — Returns candidate item IDs

**Edge cases handled:** Non-existent users (returns empty set), no unrated items left (returns empty set), fewer neighbors available than requested (uses all available).

### scorer.py
**What it does:** Scores candidate items using similarity-weighted averaging. Items rated highly by more similar users get higher scores.

**Key methods:**
- `score_candidates(...) -> list[tuple]` — Returns (item_id, score) tuples sorted by score
- `get_top_n(scored_items, n=3) -> list` — Extracts top N items

**Edge cases handled:** Empty candidates (returns empty list), no neighbors who rated an item (score = 0.0), fewer items than requested (returns all available).

### evaluator.py
**What it does:** Evaluates recommendation quality by comparing against ground-truth relevant items using the Precision@K metric.

**Key methods:**
- `precision_at_k(recommended_items, relevant_items, k=None) -> float` — Computes metric
- `evaluate(...) -> dict` — Returns full evaluation metrics

**Edge cases handled:** Empty relevant set (precision = 0.0), empty recommendations (precision = 0.0), k larger than available items (penalizes shortage).

## Running the Project

### Run the Full Pipeline
```bash
python main.py
```
This generates recommendations for all users and evaluates one user's recommendations against ground truth.

### Run Individual Components (Standalone)
Each module is independently testable and includes a `__main__` block:

```bash
python similarity.py          # Demo cosine & Jaccard similarity
python candidate_generator.py # Demo collaborative filtering
python scorer.py              # Demo scoring and ranking
python evaluator.py           # Demo evaluation metrics
```

### Sample Output
When you run `main.py`, you'll see output like:

```
=== Recommendations for alice ===
Current ratings: {'item1': 5, 'item2': 3, 'item3': 0, 'item4': 1}

Top recommendations:
  1. item3 (score: 5.00)

Evaluation Results:
  precision_at_k: 0.3333
  k: 3
  num_relevant_found: 1
  num_recommended: 1
  num_relevant_total: 1
```

## Sample Data

The project includes `sample_data.py` with mock data:

**user_ratings** — A dictionary mapping user IDs to their rating vectors:
```python
{
    "alice": {"item1": 5, "item2": 3, "item3": 0, "item4": 1},
    "bob":   {"item1": 4, "item2": 0, "item3": 0, "item4": 1},
    "carol": {"item1": 1, "item2": 1, "item3": 5, "item4": 4},
}
```

**item_tags** — A dictionary mapping item IDs to sets of tags/genres:
```python
{
    "item1": {"action", "sci-fi"},
    "item2": {"romance", "drama"},
    "item3": {"comedy"},
    "item4": {"action", "comedy"},
}
```

Users rate items on a 0–5 scale (0 = not watched/rated). Tags can be used for content-based filtering in future enhancements.

## Limitations & Next Steps

This is a teaching project. Here are ideas to extend it:

- **Real Datasets** — Use MovieLens, Amazon reviews, or Book Crossing data instead of toy data
- **Content-Based Filtering** — Weight recommendations by item tag similarity (jaccard_similarity) in addition to collaborative signals
- **Hybrid Scoring** — Combine collaborative filtering scores (70%) with content-based scores (30%)
- **Train/Test Split** — Properly evaluate by hiding some ratings during recommendation, then checking if hidden items are recommended
- **Matrix Factorization** — Implement SVD or NMF for better handling of sparse rating matrices
- **Diversity** — Add penalization for redundant recommendations (don't recommend 3 sci-fi movies if user wants variety)
- **Cold Start** — Handle new users with no ratings or new items with no ratings (popularity baseline)
- **Threshold Tuning** — Experiment with rating threshold (>= 4) and number of neighbors (top_n_neighbors) to optimize precision
- **A/B Testing** — Compare different algorithms (user-user vs. item-item vs. hybrid) on real user engagement metrics

## Scalability Notes

The current similarity computation compares every user against every other user,
making it O(n²) in the number of users — fine for small demo datasets like this one,
but it won't scale to a real platform with millions of users. In production, this
would typically be replaced with:
- Matrix factorization (e.g. SVD, ALS) to learn compact user/item embeddings
  instead of computing pairwise similarity directly
- Approximate nearest-neighbor search (e.g. Locality-Sensitive Hashing / LSH) to
  find similar users without comparing against everyone
- Precomputed similarity indices, refreshed on a schedule rather than computed
  live per request

This project prioritizes clarity and correctness of the core algorithm over
production-scale performance.

