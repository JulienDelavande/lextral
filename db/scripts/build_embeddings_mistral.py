import os
import time
from typing import Iterable, List

import psycopg2 as psycopg
from psycopg2.extras import execute_values
from datasets import load_dataset
from mistralai import Mistral
from tqdm import tqdm


MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mistral-embed")
SPLIT = os.getenv("SPLIT", "train")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))

DB_TYPE= os.getenv("DB_TYPE")
DB_PILOT=os.getenv("DB_PILOT")
DB_USER=os.getenv("DB_USER")
DB_HOST=os.getenv("DB_HOST")
DB_PORT=os.getenv("DB_PORT")
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME= os.getenv("DB_NAME")
if DB_PASSWORD:
    DB_URL = f"{DB_TYPE}://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_URL = f"{DB_TYPE}://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


assert DB_URL, "DB_URL missing"
assert MISTRAL_API_KEY, "MISTRAL_API_KEY missing"

client = Mistral(api_key=MISTRAL_API_KEY)

def chunks(xs: List, n: int) -> Iterable[List]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]

def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Appelle l'API embeddings en gérant les retries exponentiels.
    """
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            sleep_s = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(sleep_s)

def to_vector_literal(vec: List[float]) -> str:
    """
    Format compatible pgvector: '[v1, v2, ...]'.
    On passera ce string avec un cast ::vector côté SQL.
    """
    return "[" + ", ".join(f"{x:.8f}" for x in vec) + "]"

def main():
    # Charge LEDGAR (lex_glue)
    ds = load_dataset("lex_glue", "ledgar", split=SPLIT)
    label_names = ds.features["label"].names 
    # first 100
    ds = ds[30000:60000]

    print(f"Loaded {len(ds)} examples from split='{SPLIT}'")

    texts = [ex for ex in ds["text"]]
    label_ids = [int(ex) for ex in ds["label"]]
    label_txts = [label_names[lid] for lid in label_ids]

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            for idxs in tqdm(list(chunks(list(range(len(texts))), BATCH_SIZE)), desc="Embedding"):
                batch_texts = [texts[i] for i in idxs]
                batch_labels_id = [label_ids[i] for i in idxs]
                batch_labels_txt = [label_txts[i] for i in idxs]

                embs = embed_batch(batch_texts)  # List[List[float]]
                rows = []
                for lid, ltxt, t, emb in zip(batch_labels_id, batch_labels_txt, batch_texts, embs):
                    vec_literal = to_vector_literal(emb)  # ex: "[0.123, -0.456, ...]"
                    rows.append((
                        SPLIT,
                        lid,
                        ltxt,
                        t,
                        vec_literal,
                    ))

                execute_values(
                    cur,
                    """
                    INSERT INTO clause_embeddings (split, label_id, label_text, text, embedding)
                    VALUES %s
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s::vector)"
                )
        conn.commit()

    print("Done.")

if __name__ == "__main__":
    main()
