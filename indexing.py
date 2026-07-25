"""
Run this once (and again whenever the corpus or embedding model changes) to
build the Elasticsearch index: embeds every document with a free local
HuggingFace model, caches the embedding matrix to disk, creates an ES index
with both a `text` field (BM25) and an `embedding` field (dense_vector,
cosine similarity), and bulk-loads everything in.

Usage:
    python indexing.py
"""

import numpy as np
import pandas as pd
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, INDEX_DIR, INDEX_NAME, MODEL_NAME, EMBEDDING_DIMS
from esClient import es


def load_corpus() -> pd.DataFrame:
    """Load the FiQA corpus parquet file."""
    corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
    print(f"Loaded {len(corpus)} documents")
    return corpus


def embed_corpus(doc_texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Embed the full corpus, using a cached .npy file if one already exists
    so re-runs don't re-embed 57k+ docs unnecessarily."""
    embeddings_path = INDEX_DIR / "embeddings.npy"
    if embeddings_path.exists():
        print(f"Loading cached embeddings from {embeddings_path}")
        return np.load(embeddings_path)

    print(f"Embedding {len(doc_texts)} docs locally with {MODEL_NAME} (free, no API cost)")
    embeddings = model.encode(
        doc_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(embeddings_path, embeddings)
    return embeddings


def create_index():
    """Create (or recreate) the ES index with BM25 + dense_vector mapping."""
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


def bulk_index(doc_ids: list, doc_texts: list[str], doc_embeddings: np.ndarray):
    """Bulk-load documents (text + embedding together) into the ES index."""
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
    success, errors = bulk(
        es.options(request_timeout=120),
        doc_generator(),
        chunk_size=500,
    )
    print(f"Indexed: {success} docs | Errors: {len(errors) if isinstance(errors, list) else errors}")

    es.indices.refresh(index=INDEX_NAME)


def main():
    corpus = load_corpus()
    doc_ids = corpus["_id"].tolist()
    # Some FiQA docs have blank text; swap in a placeholder so embeddings
    # don't fail and row order stays aligned with doc_ids.
    doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]

    model = SentenceTransformer(MODEL_NAME)
    doc_embeddings = embed_corpus(doc_texts, model)

    create_index()
    bulk_index(doc_ids, doc_texts, doc_embeddings)

    print("Indexing complete.")


if __name__ == "__main__":
    main()