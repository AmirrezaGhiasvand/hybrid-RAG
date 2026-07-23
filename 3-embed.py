"""
Hybrid retrieval: BM25 + dense vectors, both stored and queried in
Elasticsearch. No separate vector database needed -- ES handles text search
(BM25) and approximate kNN vector search in the same index.

Pipeline:
  1. Load corpus, embed every document once with a free local HuggingFace
     model (all-MiniLM-L6-v2), cache the embedding matrix to disk.
  2. Create an ES index with both a `text` field (BM25) and an `embedding`
     field (dense_vector, cosine similarity).
  3. Bulk-index documents with their text + embedding together.
  4. Query three ways: BM25 only, kNN only, or combined hybrid.

"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import urllib3
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

# Silence "InsecureRequestWarning" -- we're intentionally skipping cert
# verification for a local self-signed certificate (dev only).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# CONFIG
DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_DIR = Path(__file__).parent / "indexes" / "dense_hf"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME = "fiqa-hybrid"
ES_HOST = "https://localhost:9200"  # https, since security is enabled
ES_USER = "elastic"
ES_PASSWORD = os.environ.get("ES_PASSWORD")
if not ES_PASSWORD:
    raise ValueError("ES_PASSWORD environment variable is not set")

MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, 384-dim
EMBEDDING_DIMS = 384

# CONNECT TO ELASTICSEARCH
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,  # local self-signed cert, dev only
)

if not es.ping():
    raise ConnectionError(f"Could not connect to Elasticsearch at {ES_HOST}")

print("Connected to Elasticsearch")

# LOAD CORPUS
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()

# Some FiQA docs have blank text; swap in a placeholder so embeddings don't
# fail and row order stays aligned with doc_ids.
doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]

print(f"Loaded {len(doc_texts)} documents")

# EMBED THE CORPUS (cached to disk after first run)
model = SentenceTransformer(MODEL_NAME)

embeddings_path = INDEX_DIR / "embeddings.npy"
if embeddings_path.exists():
    print(f"Loading cached embeddings from {embeddings_path}")
    doc_embeddings = np.load(embeddings_path)
else:
    print(f"Embedding {len(doc_texts)} docs locally with {MODEL_NAME} (free, no API cost)")
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(embeddings_path, doc_embeddings)

# CREATE INDEX: BM25 text field + dense_vector field
index_settings = {
    "settings": {
        "similarity": {
            "default": {"type": "BM25", "k1": 1.2, "b": 0.75}
        },
        "analysis": {
            "analyzer": {"default": {"type": "english"}}
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "similarity": "default"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}

if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
    print(f"Deleted existing index '{INDEX_NAME}'")

es.indices.create(index=INDEX_NAME, body=index_settings)
print(f"Created index '{INDEX_NAME}' with BM25 + dense_vector mapping")

# BULK INDEX: text + embedding together
def doc_generator():
    for doc_id, text, vec in zip(doc_ids, doc_texts, doc_embeddings):
        yield {
            "_index": INDEX_NAME,
            "_id": str(doc_id),
            "_source": {
                "text": text,
                "embedding": vec.tolist(),  # numpy array -> plain list for JSON
            },
        }

print(f"Indexing {len(doc_texts)} documents (text + embeddings)...")
success, errors = bulk(es, doc_generator(), chunk_size=500, request_timeout=120)
print(f"Indexed: {success} docs | Errors: {len(errors) if isinstance(errors, list) else errors}")

es.indices.refresh(index=INDEX_NAME)

# SEARCH FUNCTIONS
def embed_query(query: str) -> list[float]:
    vec = model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    return vec.tolist()


def search_bm25(query: str, top_k: int = 10):
    """Keyword search only -- best for exact terms, symbols, codes."""
    resp = es.search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        size=top_k,
    )
    return [(h["_id"], h["_score"], h["_source"]["text"]) for h in resp["hits"]["hits"]]


def search_dense(query: str, top_k: int = 10):
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
    return [(h["_id"], h["_score"], h["_source"]["text"]) for h in resp["hits"]["hits"]]


def search_hybrid_boost(query: str, top_k: int = 10, vector_boost: float = 0.5):
    """BM25 + kNN combined via raw score blending -- ES adds both scores.
    Note: BM25 scores are unbounded, cosine scores sit in [0, 1], so this
    blend is scale-mismatched. See search_hybrid_rrf for the better fix."""
    resp = es.search(
        index=INDEX_NAME,
        query={"match": {"text": query}},
        knn={
            "field": "embedding",
            "query_vector": embed_query(query),
            "k": top_k,
            "num_candidates": max(100, top_k * 10),
            "boost": vector_boost,
        },
        size=top_k,
    )
    return [(h["_id"], h["_score"], h["_source"]["text"]) for h in resp["hits"]["hits"]]


# RECIPROCAL RANK FUSION (RRF)
# Fuse RANKINGS instead of raw scores. BM25 scores are unbounded while
# cosine similarities sit in [0, 1] -- averaging them directly is comparing
# apples to oranges. RRF only cares about each doc's rank position in each
# retriever's list, which sidesteps the scale mismatch entirely.
#
#   rrf_score(d) = sum over each retriever r of 1 / (k + rank_r(d))
#
# k is a smoothing constant, conventionally 60 (Cormack et al., 2009:
# https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

K_RRF = 60


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = K_RRF
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into one ranked list."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def search_hybrid_rrf(query: str, top_k: int = 10, candidate_k: int = 50):
    """Retrieve top candidate_k from BM25 and dense separately, fuse by
    rank (RRF), return top_k. Two ES calls + local fusion, but scale-safe."""
    bm25_ids = [doc_id for doc_id, _, _ in search_bm25(query, top_k=candidate_k)]
    dense_ids = [doc_id for doc_id, _, _ in search_dense(query, top_k=candidate_k)]
    fused = reciprocal_rank_fusion([bm25_ids, dense_ids])[:top_k]

    # Attach text for display/debugging, same shape as the other search_* fns
    results = []
    for doc_id, score in fused:
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        results.append((doc_id, score, text))
    return results


# EXAMPLE QUERIES
if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"

    for label, fn in [
        ("BM25 only", search_bm25),
        ("Dense only", search_dense),
        ("Hybrid (score boost)", search_hybrid_boost),
        ("Hybrid (RRF)", search_hybrid_rrf),
    ]:
        print(f"\n=== {label} ===")
        for rank, (doc_id, score, text) in enumerate(fn(query, top_k=5), start=1):
            snippet = text[:150].replace("\n", " ")
            print(f"{rank}. [{doc_id}] score={score:.4f}\n   {snippet}...\n")