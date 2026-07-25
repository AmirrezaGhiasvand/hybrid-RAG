"""
Reranking: take the union of top candidates from BM25 and dense search
directly, and reorder them with a cross-encoder. Return the top-10.

A bi-encoder (the dense retriever in retrievers.py) embeds the query and the
document separately, then compares with cosine. A cross-encoder feeds the
query and document INTO the same model together and returns a single
relevance score. Cross-encoders are slower (one forward pass per candidate,
not one embedding total), but the joint attention catches subtleties that
two independent embeddings miss.

Candidates come from BM25 + dense directly (unioned, deduped), not from
RRF's fused output -- RRF already truncates to its own top-k based on rank
position, which can drop a document one retriever loved but the other
missed. Feeding the reranker the raw union avoids losing that document
before the cross-encoder even sees it.

This version uses a free, local, open-source cross-encoder via
sentence-transformers -- no API key, no per-call cost, runs on CPU.

Model options (swap RERANK_MODEL to try):
  - cross-encoder/ms-marco-MiniLM-L-6-v2   fast, small (~22M params), solid baseline
  - BAAI/bge-reranker-base                  stronger quality, similar size class
  - BAAI/bge-reranker-v2-m3                 strongest, multilingual, heavier/slower

More info: https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
"""

from sentence_transformers import CrossEncoder

from retrievers import search_bm25, search_dense, search_hybrid_rrf

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Step 1: Load the cross-encoder (downloads weights on first run, then cached
# locally under ~/.cache/huggingface -- no internet needed after that).
# ---------------------------------------------------------------------------
cross_encoder = CrossEncoder(RERANK_MODEL)


# ---------------------------------------------------------------------------
# Step 2: Rerank with the cross-encoder
# ---------------------------------------------------------------------------
def search_reranked(
    query: str, top_k: int = 10, candidate_k: int = 50
) -> list[tuple[str, float, str]]:
    """Retrieve candidate_k results from BM25 and dense independently, union
    them (deduped by doc_id), then rerank the full pool with the
    cross-encoder and return the top_k.

    This differs from reranking RRF's output: RRF already fuses/truncates
    to its own top candidate_k based on rank position, which can silently
    drop a document that one retriever ranked very highly but the other
    missed entirely. Pulling directly from BM25 + dense gives the
    cross-encoder the full union each retriever independently believed in,
    before any fusion logic has a chance to cut something.
    """
    bm25_results = search_bm25(query, top_k=candidate_k)
    dense_results = search_dense(query, top_k=candidate_k)

    # Union by doc_id, deduping candidates that both retrievers surfaced.
    candidates: dict[str, str] = {}
    for doc_id, _, text in bm25_results:
        candidates[doc_id] = text
    for doc_id, _, text in dense_results:
        candidates[doc_id] = text

    candidate_ids = list(candidates.keys())
    candidate_texts = list(candidates.values())

    # Cross-encoder scores each (query, document) pair jointly -- this is
    # the O(len(candidates)) forward-pass cost that makes reranking too slow
    # to run over the full corpus, but fine over a shortlist of up to ~100.
    pairs = [[query, text] for text in candidate_texts]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, scores, candidate_texts), key=lambda x: -x[1]
    )[:top_k]
    return [(doc_id, float(score), text) for doc_id, score, text in ranked]


# ---------------------------------------------------------------------------
# Step 3: Compare hybrid vs hybrid + rerank
# ---------------------------------------------------------------------------
def show(label: str, results: list[tuple[str, float, str]]) -> None:
    print(f"\n{label}")
    for i, (doc_id, score, text) in enumerate(results[:5], 1):
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    show("Hybrid (RRF) only", search_hybrid_rrf(query, top_k=5))
    show(f"Hybrid + {RERANK_MODEL}", search_reranked(query, top_k=5))