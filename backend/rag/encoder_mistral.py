import os
import time
from typing import List, Iterable
from mistralai import Mistral

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mistral-embed")
EMBED_DIM = 1024

_client = None

def _get_client() -> Mistral:
    global _client
    if _client is None:
        assert MISTRAL_API_KEY, "MISTRAL_API_KEY missing"
        _client = Mistral(api_key=MISTRAL_API_KEY)
    return _client

def _chunks(xs: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]

def embed_texts(texts: List[str], batch_size: int = 64, max_retries: int = 5) -> List[List[float]]:
    client = _get_client()
    vectors: List[List[float]] = []
    for batch in _chunks(texts, batch_size):
        attempt = 0
        while True:
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
                batch_vecs = [d.embedding for d in resp.data]
                for v in batch_vecs:
                    if len(v) != EMBED_DIM:
                        raise ValueError(f"Embedding dim {len(v)} != {EMBED_DIM}")
                vectors.extend(batch_vecs)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                time.sleep(0.5 * (2 ** (attempt - 1)))
    return vectors
