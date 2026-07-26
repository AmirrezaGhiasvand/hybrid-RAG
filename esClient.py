"""
Single shared Elasticsearch client. Both indexing.py and retrievers.py import
`es` from here so there's exactly one connection setup in the whole project.
"""

import urllib3
from elasticsearch import Elasticsearch

from config import ES_HOST, ES_PASSWORD, ES_USER

# Silence "InsecureRequestWarning" -- we're intentionally skipping cert
# verification for a local self-signed certificate (dev only).
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, str(ES_PASSWORD)),
    verify_certs=False,  # local self-signed cert, dev only
)

if not es.ping():
    raise ConnectionError(f"Could not connect to Elasticsearch at {ES_HOST}")
