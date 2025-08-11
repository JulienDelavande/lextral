# backend/rag/retriever_pgvector.py
import os
from typing import List, Optional, Tuple
import psycopg2

DB_TYPE  = os.getenv("DB_TYPE", "postgresql")
DB_USER  = os.getenv("DB_USER", "postgres")
DB_PASS  = os.getenv("DB_PASSWORD")
DB_HOST  = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT  = os.getenv("DB_PORT", "5432")
DB_NAME  = os.getenv("DB_NAME", "lextral-db")

if DB_PASS:
    DB_URL = f"{DB_TYPE}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DB_URL = f"{DB_TYPE}://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
print(f"Using DB_URL: {DB_URL}")

def _to_vector_literal(vec: List[float]) -> str:
    return "[" + ", ".join(f"{x:.8f}" for x in vec) + "]"

def search_similar(
    text_embedding: List[float],
    top_k: int = 5,
    min_sim: Optional[float] = None,
    split: Optional[str] = "train",
    label_id: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """
    Retourne: [(label_text, text, similarity), ...] trié par similarité décroissante.
    Similarité = 1 - (embedding <=> query_vec)  avec opérateur cosinus de pgvector.
    """
    vec_lit = _to_vector_literal(text_embedding)

    # On calcule 'sim' une seule fois dans la CTE q, puis on filtre/ordonne dessus.
    sql = """
    WITH q AS (
      SELECT
        label_text,
        text,
        1.0 - (embedding <=> %s::vector) AS sim
      FROM clause_embeddings
      WHERE 1=1
    """
    params = [vec_lit]

    if split is not None:
        sql += " AND split = %s"
        params.append(split)
    if label_id is not None:
        sql += " AND label_id = %s"
        params.append(label_id)

    sql += """
    )
    SELECT label_text, text, sim
    FROM q
    """
    if min_sim is not None:
        sql += " WHERE sim >= %s"
        params.append(float(min_sim))

    sql += " ORDER BY sim DESC LIMIT %s"
    params.append(int(top_k))

    # Exécution
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [(r[0], r[1], float(r[2])) for r in rows]
