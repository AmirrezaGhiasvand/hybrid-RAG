"""
Search functions for the hybrid retrieval project. Assumes indexing.py has
already been run and the ES index exists. Import these into any script or
notebook without re-embedding or re-indexing anything.

    from retrievers import search_bm25, search_dense, search_hybrid_boost, search_hybrid_rrf
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_NAME, MODEL_NAME, K_RRF
from esClient import es

# Loaded once at import time, reused across all search calls.
model = SentenceTransformer(MODEL_NAME)
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")


def embed_query(query: str) -> list[float]:
    vec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    return vec.tolist()


def normalize_scores(
    results: list[tuple[str, float, str]]
) -> list[tuple[str, float, str]]:
    """Min-max normalize scores within a result list to [0, 1].

    This does NOT make scores comparable across different queries or corpora
    -- it only rescales one ranked list so its best doc = 1.0 and worst = 0.0.
    That's enough to (a) let you eyeball BM25/dense/RRF results side by side
    on a common scale, and (b) safely blend BM25 + dense scores without one
    silently dominating the other (see search_hybrid_boost below).
    """
    if not results:
        return results
    scores = [s for _, s, _ in results]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:  # all scores equal, avoid divide-by-zero
        return [(doc_id, 1.0, text) for doc_id, _, text in results]
    return [(doc_id, (s - lo) / (hi - lo), text) for doc_id, s, text in results]


def search_bm25(query: str, top_k: int = 10, normalize: bool = False):
    """Keyword search only -- best for exact terms, symbols, codes."""
    resp = es.search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        size=top_k,
    )
    results = [(h["_id"], h["_score"], h["_source"]["text"]) for h in resp["hits"]["hits"]]
    return normalize_scores(results) if normalize else results


def search_dense(query: str, top_k: int = 10, normalize: bool = False):
    """Vector search only -- best for paraphrase, semantic similarity."""
    resp = es.search(
        index=INDEX_NAME,
        knn={
            "field": "embedding",
            "query_vector": embed_query(query),
            "k": top_k,
            "num_candidates": max(100, top_k * 10),
        },
        size=top_k,
    )
    results = [(h["_id"], h["_score"], h["_source"]["text"]) for h in resp["hits"]["hits"]]
    return normalize_scores(results) if normalize else results


def search_hybrid_boost(query: str, top_k: int = 10, candidate_k: int = 50, weight: float = 0.5):
    """BM25 + dense combined by normalizing each to [0, 1] first, then
    blending with a weighted average. This avoids the naive-boost bug: raw
    BM25 scores (unbounded, often 15-30+) completely dwarf raw cosine scores
    (capped at 1.0), so a small boost multiplier on the vector score has
    almost no effect -- the result silently collapses to BM25-only.
    Normalizing both to the same [0, 1] scale before blending means `weight`
    actually controls the balance as intended."""
    bm25_results = normalize_scores(search_bm25(query, top_k=candidate_k))
    dense_results = normalize_scores(search_dense(query, top_k=candidate_k))

    combined: dict[str, float] = defaultdict(float)
    texts: dict[str, str] = {}
    for doc_id, score, text in bm25_results:
        combined[doc_id] += (1 - weight) * score
        texts[doc_id] = text
    for doc_id, score, text in dense_results:
        combined[doc_id] += weight * score
        texts[doc_id] = text

    ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    return [(doc_id, score, texts[doc_id]) for doc_id, score in ranked]


# ---------------------------------------------------------------------------
# RECIPROCAL RANK FUSION (RRF)
# ---------------------------------------------------------------------------
# Fuse RANKINGS instead of raw scores. BM25 scores are unbounded while
# cosine similarities sit in [0, 1] -- averaging them directly is comparing
# apples to oranges. RRF only cares about each doc's rank position in each
# retriever's list, which sidesteps the scale mismatch entirely.
#
#   rrf_score(d) = sum over each retriever r of 1 / (k + rank_r(d))
#
# k is a smoothing constant, conventionally 60 (Cormack et al., 2009:
# https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = K_RRF
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into one ranked list."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def search_hybrid_rrf(query: str, top_k: int = 10, candidate_k: int = 50, normalize: bool = False):
    """Retrieve top candidate_k from BM25 and dense separately, fuse by
    rank (RRF), return top_k. Two ES calls + local fusion, but scale-safe
    by construction -- RRF never touches raw scores, only rank position."""
    bm25_ids = [doc_id for doc_id, _, _ in search_bm25(query, top_k=candidate_k)]
    dense_ids = [doc_id for doc_id, _, _ in search_dense(query, top_k=candidate_k)]
    fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:top_k]

    # Attach text for display/debugging, same shape as the other search_* fns
    results = []
    for doc_id, score in fused:
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        results.append((doc_id, score, text))
    return normalize_scores(results) if normalize else results