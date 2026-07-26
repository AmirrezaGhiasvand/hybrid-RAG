"""
Reranking: take the top candidates from hybrid retrieval and reorder them
with a cross-encoder. Return the top-10.

A bi-encoder (the dense retriever in retrievers.py) embeds the query and the
document separately, then compares with cosine. A cross-encoder feeds the
query and document INTO the same model together and returns a single
relevance score. Cross-encoders are slower (one forward pass per candidate,
not one embedding total), but the joint attention catches subtleties that
two independent embeddings miss.

Uses a free, local, open-source cross-encoder via
sentence-transformer

Model options (swap RERANK_MODEL to try):
  - cross-encoder/ms-marco-MiniLM-L-6-v2   fast, small (~22M params), solid baseline
  - BAAI/bge-reranker-base                  stronger quality, similar size class
  - BAAI/bge-reranker-v2-m3                 strongest, multilingual, heavier/slower

"""

from sentence_transformers import CrossEncoder

from retrievers import search_hybrid_boost, search_hybrid_rrf

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Load the cross-encoder (downloads weights on first run, then cached
# ---------------------------------------------------------------------------
cross_encoder = CrossEncoder(RERANK_MODEL)


def _rerank(
    query: str, candidates: list[tuple[str, float, str]], top_k: int
) -> list[tuple[str, float, str]]:
    """Shared rerank logic: score every candidate jointly with the query
    using the cross-encoder, return the top_k re-sorted by that score."""
    candidate_ids = [doc_id for doc_id, _, _ in candidates]
    candidate_texts = [text for _, _, text in candidates]

    pairs = [[query, text] for text in candidate_texts]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(candidate_ids, scores, candidate_texts), key=lambda x: -x[1])[
        :top_k
    ]
    return [(doc_id, float(score), text) for doc_id, score, text in ranked]


# ---------------------------------------------------------------------------
# Rerank on top of RRF hybrid
# ---------------------------------------------------------------------------
def search_reranked(
    query: str, top_k: int = 10, candidate_k: int = 50
) -> list[tuple[str, float, str]]:
    """Retrieve candidate_k results via hybrid RRF, then rerank all of them
    with the cross-encoder and return the top_k."""
    candidates = search_hybrid_rrf(query, top_k=candidate_k, candidate_k=candidate_k)
    return _rerank(query, candidates, top_k)


# ---------------------------------------------------------------------------
# Rerank on top of the boost (normalized blend) hybrid instead of RRF
# ---------------------------------------------------------------------------
def search_reranked_boost(
    query: str, top_k: int = 10, candidate_k: int = 50
) -> list[tuple[str, float, str]]:
    """Retrieve candidate_k results via the normalized-blend hybrid
    (search_hybrid_boost), then rerank all of them with the cross-encoder
    and return the top_k. Same idea as search_reranked, just sourcing
    candidates from the boost hybrid's ranking instead of RRF's."""
    candidates = search_hybrid_boost(query, top_k=candidate_k, candidate_k=candidate_k)
    return _rerank(query, candidates, top_k)


# ---------------------------------------------------------------------------
# Compare RRF vs Boost rerank
# ---------------------------------------------------------------------------
def show(label: str, results: list[tuple[str, float, str]]) -> None:
    print(f"\n{label}")
    for i, (doc_id, score, text) in enumerate(results[:5], 1):
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    show("Hybrid (RRF)", search_hybrid_rrf(query, top_k=5))
    show(f"Hybrid (RRF) Reranked + {RERANK_MODEL}", search_reranked(query, top_k=5))
    show("Hybrid (Boost)", search_hybrid_boost(query, top_k=5))
    show(
        f"Hybrid (Boost) Reranked + {RERANK_MODEL}",
        search_reranked_boost(query, top_k=5),
    )
