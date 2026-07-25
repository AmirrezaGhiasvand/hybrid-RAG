"""
The payoff. NDCG@10 across BM25, dense, hybrid (RRF), and hybrid + rerank,
on the FiQA-2018 test set.

Normalized Discounted Cumulative Gain at 10 (NDCG@10) is the standard
retrieval metric. It rewards putting relevant docs high in the top-10 and
penalizes putting them low. Perfect ranking = 1.0.

Since the reranker here is a free local cross-encoder (not a rate-limited
API), there's no free-tier constraint forcing a small sample -- but a full
sweep over all test queries still costs real time on CPU (one cross-encoder
forward pass per candidate per query). EVAL_SAMPLE_SIZE defaults to 50 for a
fast sanity check; set it to None to run the entire test set.

Public BEIR baselines for FiQA NDCG@10:
    BM25                       ~24
    text-embedding-3-small     ~31
    + cross-encoder rerank     ~40+

The wide spread between BM25 and reranked hybrid is the point: BM25 alone
struggles on paraphrase-heavy questions, and every stage of the pipeline
earns its keep.

"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR
from retrievers import search_bm25, search_dense, search_hybrid_rrf, search_hybrid_boost
from rerank import search_reranked

EVAL_SAMPLE_SIZE = None  # set to None to run the full test set
SEED = 42


# ---------------------------------------------------------------------------
# Step 1: NDCG@k in pure numpy/math
# ---------------------------------------------------------------------------
def ndcg_at_k(predicted_ids: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Normalized discounted cumulative gain for a single query."""
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(predicted_ids[:k])
    )
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Step 2: Load queries + ground truth, pick the sample
# ---------------------------------------------------------------------------
queries = pd.read_parquet(DATA_DIR / "queries.parquet")
qrels_df = pd.read_parquet(DATA_DIR / "qrels.parquet")

qrels: dict[str, dict[str, int]] = defaultdict(dict)
for _, row in qrels_df.iterrows():
    qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

queries_with_qrels = queries[queries["_id"].astype(str).isin(qrels.keys())].copy()

if EVAL_SAMPLE_SIZE is not None:
    sample = queries_with_qrels.sample(n=EVAL_SAMPLE_SIZE, random_state=SEED)
else:
    sample = queries_with_qrels

print(f"Evaluating on {len(sample)} queries (sampled from {len(queries_with_qrels)})")


# ---------------------------------------------------------------------------
# Step 3: Score every method on the sample
# ---------------------------------------------------------------------------
def run_evaluation() -> dict[str, list[float]]:
    results: dict[str, list[float]] = defaultdict(list)

    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Evaluating"):
        query_id = str(row["_id"])
        query_text = row["text"]
        relevant = qrels[query_id]

        bm25_ids = [doc_id for doc_id, _, _ in search_bm25(query_text, top_k=10)]
        dense_ids = [doc_id for doc_id, _, _ in search_dense(query_text, top_k=10)]
        hybrid_ids = [doc_id for doc_id, _, _ in search_hybrid_rrf(query_text, top_k=10)]
        reranked_ids = [doc_id for doc_id, _, _ in search_reranked(query_text, top_k=10)]
        boost_ids = [doc_id for doc_id, _, _ in search_hybrid_boost(query_text, top_k=10)]

        results["BM25"].append(ndcg_at_k(bm25_ids, relevant))
        results["Dense"].append(ndcg_at_k(dense_ids, relevant))
        results["Hybrid (RRF)"].append(ndcg_at_k(hybrid_ids, relevant))
        results["Hybrid + Rerank"].append(ndcg_at_k(reranked_ids, relevant))
        results["Boost Hybrid"].append(ndcg_at_k(boost_ids, relevant))

    return results


# ---------------------------------------------------------------------------
# Step 4: Print the table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_evaluation()
    print(f"\nNDCG@10 x100 on FiQA ({len(sample)} sampled test queries)")
    print("-" * 42)
    for method, scores in results.items():
        print(f"  {method:<22} {np.mean(scores) * 100:.2f}")