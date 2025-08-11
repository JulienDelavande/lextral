CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS clause_embeddings (
  id         BIGSERIAL PRIMARY KEY,
  split      TEXT NOT NULL,                 -- 'train' / 'validation' / 'test'
  label_id   INT  NOT NULL,
  label_text TEXT NOT NULL,
  text       TEXT NOT NULL,
  embedding  VECTOR(1024) NOT NULL,         --  dim=1024
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Index ANN HNSW (cosine)
CREATE INDEX IF NOT EXISTS idx_clause_embeddings_ann
ON clause_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_clause_embeddings_label
ON clause_embeddings (label_id);
