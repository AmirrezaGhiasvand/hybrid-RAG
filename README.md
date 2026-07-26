# Hybrid RAG Retrieval — FiQA

A retrieval pipeline built and evaluated on the FiQA-2018 financial Q&A dataset (via BEIR), comparing sparse (BM25), dense, hybrid fusion, and cross-encoder reranking approaches — all backed by a single Elasticsearch index.

## Architecture

Elasticsearch stores both the raw text (BM25-indexed) and dense embeddings (`dense_vector`, cosine similarity) for every document in one index — no separate vector database needed. Embeddings come from a free, local `sentence-transformers` model (`all-MiniLM-L6-v2`), and reranking uses a free, local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Nothing in this pipeline requires a paid API.

```
exploreData.ipynb # get to know the FiQA dataset more
config.py         # shared constants (paths, ES connection info, model names)
esClient.py       # single Elasticsearch client, imported everywhere else
indexing.py         # one-time: embed corpus, create ES index, bulk-load
retrievers.py         # search_bm25, search_dense, search_hybrid_boost, search_hybrid_rrf
rerank.py               # cross-encoder reranking on top of either hybrid method
eval.py                   # NDCG@10 evaluation across all methods on the FiQA test set
```

## Methods compared

- **BM25** — Elasticsearch's built-in keyword search.
- **Dense** — cosine similarity over `all-MiniLM-L6-v2` embeddings, computed via Elasticsearch's native kNN search.
- **Hybrid (RRF)** — BM25 and dense retrieved independently, fused by Reciprocal Rank Fusion (rank-based, ignores raw score magnitude).
- **Hybrid (Boost)** — BM25 and dense scores min-max normalized to `[0, 1]` independently, then combined with a weighted average.
- **+ Rerank** — either hybrid method's top candidates re-scored by a cross-encoder, which jointly attends over the query and document text rather than comparing independent embeddings.

## Results

NDCG@10 × 100, full FiQA test set (648 queries):

| Method                  | NDCG@10   |
| ----------------------- | --------- |
| BM25                    | 25.37     |
| Dense                   | 36.61     |
| Hybrid (RRF)            | 36.71     |
| Hybrid (RRF) + Rerank   | 37.24     |
| Hybrid (Boost)          | **37.74** |
| Hybrid (Boost) + Rerank | 37.33     |

**Boost Hybrid (no reranking) is the best-performing method overall.**

## Key findings

- **Dense retrieval does almost all of the work.** BM25 → Dense is a +11.24 jump; Dense → Hybrid (RRF) is only +0.10. On FiQA's natural-language questions, paraphrase-matching (dense's strength) dominates over exact-term matching (BM25's strength) — BM25 mostly isn't adding much once dense is in the mix.
- **Score-normalized blending (Boost) beat rank-based fusion (RRF)** on this dataset, contrary to the usual expectation that RRF is the safer default. With scores properly normalized to `[0, 1]` before blending (see "the boost-hybrid bug" below), the extra magnitude information Boost preserves and RRF discards ended up helping here.
- **Reranking pulled results toward its own ceiling rather than improving the strongest input.** It helped the weaker RRF hybrid (36.71 → 37.24) but hurt the stronger Boost hybrid (37.74 → 37.33) — both landed in roughly the same ~37.2–37.3 band regardless of starting point. This suggests `ms-marco-MiniLM-L-6-v2` has its own quality ceiling on this domain, below Boost Hybrid's unassisted result.
- **A 50-query sample gave the wrong answer on reranking.** An early evaluation on 50 sampled queries showed reranking as net negative; the full 648-query run reversed that conclusion for the RRF variant. Small-sample IR evaluation is noisy enough to flip conclusions — worth running the full set before trusting a verdict.
- **The naive boost-hybrid bug:** an early version blended raw BM25 scores (unbounded, often 15–30+) with raw cosine scores (capped at `[0, 1]`) using a fixed multiplier. Because of the scale mismatch, the vector score had almost no real influence — the "hybrid" result was empirically identical to BM25 alone. Fixed by normalizing both score sets to `[0, 1]` before blending.
- **~19% of the corpus exceeds the embedding model's 256-token window** and gets silently truncated during embedding (no warning, unlike the cross-encoder's truncation). Despite this, Dense still outperforms the published `text-embedding-3-small` BEIR baseline (~31) at 36.61 — so truncation is a real, measurable limitation but not currently a blocking one.

## Running it

Requires a local Elasticsearch instance (security enabled, `ES_PASSWORD` set as an environment variable) and Python packages: all listed on requirements.txt
