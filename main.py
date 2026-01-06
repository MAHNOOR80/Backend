# FULL UPDATED FILE USING FASTEMBED (NO API KEY REQUIRED)
# Cohere COMPLETELY REMOVED

#!/usr/bin/env python3

import os
import sys
import time
import uuid
import hashlib
import logging
import argparse
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
class Config:
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "10"))
        self.max_depth = int(os.getenv("MAX_DEPTH", "2"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---------------- QDRANT ----------------
def get_qdrant_client():
    return QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key or None
    )

qdrant_client = get_qdrant_client()

# ---------------- FASTEMBED MODEL ----------------
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384

# ---------------- DATA MODELS ----------------
@dataclass
class DocumentContent:
    id: str
    url: str
    content: str
    title: str
    created_at: datetime
    chunk_index: int = 0

# ---------------- COLLECTION ----------------
def create_collection(name="rag_embedding"):
    collections = qdrant_client.get_collections().collections
    if name in [c.name for c in collections]:
        logger.info("Collection already exists")
        return

    qdrant_client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    logger.info("Collection created")

# ---------------- EMBEDDING ----------------
def embed(texts: List[str]) -> List[List[float]]:
    return [e.tolist() for e in embedding_model.embed(texts)]

# ---------------- SAVE TO QDRANT ----------------
def save_embedding(vector, doc: DocumentContent, collection="rag_embedding"):
    point = PointStruct(
        id=doc.id,
        vector=vector,
        payload={
            "url": doc.url,
            "title": doc.title,
            "content": doc.content,
            "chunk_index": doc.chunk_index,
            "created_at": doc.created_at.isoformat()
        }
    )
    qdrant_client.upsert(collection_name=collection, points=[point])

# ---------------- CHUNKING ----------------
def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + config.chunk_size
        chunks.append(text[start:end])
        start += config.chunk_size - config.chunk_overlap
    return chunks

# ---------------- CRAWLING ----------------
def is_valid_url(url):
    try:
        r = urlparse(url)
        return r.scheme and r.netloc
    except:
        return False


def crawl_site(root_url):
    visited, queue, results = set(), [(root_url, 0)], []

    while queue:
        url, depth = queue.pop(0)
        if url in visited or depth > config.max_depth:
            continue

        visited.add(url)
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            title = soup.title.text if soup.title else ""
            text = soup.get_text(" ", strip=True)

            if len(text) > 100:
                results.append({"url": url, "title": title, "content": text})

            for a in soup.find_all("a", href=True):
                link = urljoin(url, a['href'])
                if is_valid_url(link) and root_url in link:
                    queue.append((link, depth + 1))
        except Exception:
            continue

    return results

# ---------------- PIPELINE ----------------
def run_pipeline(url, collection="rag_embedding"):
    create_collection(collection)
    pages = crawl_site(url)

    for page in pages:
        chunks = chunk_text(page['content'])
        embeddings = embed(chunks)

        for i, vector in enumerate(embeddings):
            doc = DocumentContent(
                id=str(uuid.uuid4()),
                url=page['url'],
                title=page['title'],
                content=chunks[i],
                created_at=datetime.utcnow(),
                chunk_index=i
            )
            save_embedding(vector, doc, collection)

    logger.info("Pipeline completed")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--collection", default="rag_embedding")
    args = parser.parse_args()

    run_pipeline(args.url, args.collection)
