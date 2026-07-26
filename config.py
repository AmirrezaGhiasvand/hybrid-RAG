"""
Shared configuration for the hybrid BM25 + dense retrieval project.
Import from here instead of hardcoding these values in multiple files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "fiqa"
INDEX_DIR = BASE_DIR / "indexes" / "dense_hf"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------------
INDEX_NAME = "fiqa-hybrid"
ES_HOST = "https://localhost:9200"  # https, since security is enabled
ES_USER = "elastic"
ES_PASSWORD = os.environ.get("ES_PASSWORD")
if not ES_PASSWORD:
    raise ValueError("ES_PASSWORD environment variable is not set")
assert isinstance(ES_PASSWORD, str)  # narrows type for downstream imports

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, 384-dim
EMBEDDING_DIMS = 384

# RRF
K_RRF = 60  # smoothing constant, conventionally 60 (Cormack et al., 2009)
