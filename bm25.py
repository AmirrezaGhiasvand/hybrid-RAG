"""
    Run Elasticsearch (Docker example):
    docker run -d --name es -p 9200:9200 -e "discovery.type=single-node" \
        -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch
"""

from pathlib import Path
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import urllib3
import os
from dotenv import load_dotenv
load_dotenv()
# Silence "InsecureRequestWarning" since we're intentionally skipping cert
# verification for a local self-signed certificate (dev only).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# CONFIG

DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_NAME = "fiqa-bm25"
ES_HOST = "https://localhost:9200"   # note: https, not http, since security is enabled
ES_USER = "elastic"

ES_PASSWORD = os.environ.get("ES_PASSWORD") # replace with your own auto-generated password

if not ES_PASSWORD:
    raise ValueError("ES_PASSWORD environment variable is not set")

# CONNECT

es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,   # skip cert verification for local self-signed cert (dev only)
)

if not es.ping():
    raise ConnectionError(f"Could not connect to Elasticsearch at {ES_HOST}")


# LOAD CORPUS

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].to_list()
doc_texts = corpus["text"].to_list()
print(f"Loaded {len(doc_texts)} documents")


# CREATE INDEX WITH BM25 SIMILARITY

index_settings = {
    "settings": {
        "similarity": {
            "default": {
                "type": "BM25",
                "k1": 1.2,
                "b": 0.75
            }
        },
        "analysis": {
            "analyzer": {
                "default": {
                    "type": "english"  # handles lowercasing, stopwords, stemming
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "similarity": "default"
            }
        }
    }
}

# Delete index if it already exists (fresh start)
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
    print(f"Deleted existing index '{INDEX_NAME}'")

es.indices.create(index=INDEX_NAME, body=index_settings)
print(f"Created index '{INDEX_NAME}' with BM25 similarity")


# BULK INDEX DOCUMENTS

def doc_generator():
    for doc_id, text in zip(doc_ids, doc_texts):
        yield {
            "_index": INDEX_NAME,
            "_id": str(doc_id),
            "_source": {"text": text}
        }

print(f"Indexing {len(doc_texts)} documents with BM25...")
success, errors = bulk(es, doc_generator(), chunk_size=2000, request_timeout=120)
print(f"Indexed: {success} docs | Errors: {len(errors) if isinstance(errors, list) else errors}")

# Make sure documents are searchable immediately
es.indices.refresh(index=INDEX_NAME)


# SEARCH FUNCTION

def search_bm25(query, top_k=10):
    resp = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
    )
    hits = resp["hits"]["hits"]
    return [(hit["_id"], hit["_score"], hit["_source"]["text"]) for hit in hits]


# EXAMPLE QUERY

if __name__ == "__main__":
    query = "Where should i park my rainy-day fund?"
    results = search_bm25(query, top_k=5)

    print(f"\nTop results for query: '{query}'\n")
    for rank, (doc_id, score, text) in enumerate(results, start=1):
        snippet = text[:150].replace("\n", " ")
        print(f"{rank}. [{doc_id}] score={score:.4f}\n   {snippet}...\n")