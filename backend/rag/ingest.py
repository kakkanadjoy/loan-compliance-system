"""Corpus ingestion — markdown documents -> embeddings -> pgvector.

Reads every document in data/corpus, embeds it with a local
sentence-transformers model (no API keys needed), and stores text +
metadata + embedding in Postgres. Idempotent: re-running replaces the
corpus atomically, so edit rules.yaml -> regenerate -> re-ingest at will.

The table carries BOTH an embedding column (vector search) and a
generated tsvector column (keyword search) — the two halves of hybrid
retrieval in search.py.

Requires the docker compose stack: `docker compose up -d` from project root.
Connection string via DATABASE_URL env var, defaulting to the compose
stack's credentials.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://compliance:compliance@localhost:5432/compliance"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, runs locally
EMBED_DIM = 384

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_chunks (
    id                SERIAL PRIMARY KEY,
    doc_id            TEXT NOT NULL UNIQUE,
    doc_type          TEXT NOT NULL,
    rule_id           TEXT,
    section           TEXT,
    title             TEXT NOT NULL,
    exception_code    TEXT,
    severity          TEXT,
    waiver_authority  TEXT,
    applies_to        TEXT,
    body              TEXT NOT NULL,
    embedding         vector({EMBED_DIM}) NOT NULL,
    tsv               tsvector GENERATED ALWAYS AS
                        (to_tsvector('english', title || ' ' || body)) STORED
);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_tsv
    ON policy_chunks USING GIN (tsv);
"""


def parse_doc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return {
        "doc_id": meta["doc_id"],
        "doc_type": meta.get("doc_type", "rule"),
        "rule_id": meta.get("rule_id"),
        "section": meta.get("section"),
        "title": meta["title"],
        "exception_code": meta.get("exception_code"),
        "severity": meta.get("severity"),
        "waiver_authority": meta.get("waiver_authority"),
        "applies_to": meta.get("applies_to"),
        "body": body.strip(),
    }


def main() -> None:
    docs = sorted(CORPUS_DIR.glob("*.md"))
    if not docs:
        sys.exit(f"no corpus documents in {CORPUS_DIR} — run rag/generate_corpus.py first")

    print(f"loading embedding model {EMBED_MODEL} (first run downloads ~90MB)...")
    model = SentenceTransformer(EMBED_MODEL)

    parsed = [parse_doc(p) for p in docs]
    texts = [f"{d['title']}\n\n{d['body']}" for d in parsed]
    embeddings = model.encode(texts, normalize_embeddings=True)

    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
            cur.execute("DELETE FROM policy_chunks")  # atomic replace within txn
            for d, emb in zip(parsed, embeddings):
                cur.execute(
                    """
                    INSERT INTO policy_chunks
                        (doc_id, doc_type, rule_id, section, title, exception_code,
                         severity, waiver_authority, applies_to, body, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        d["doc_id"], d["doc_type"], d["rule_id"], d["section"],
                        d["title"], d["exception_code"], d["severity"],
                        d["waiver_authority"], d["applies_to"], d["body"], emb,
                    ),
                )
        conn.commit()

        with conn.cursor() as cur:
            cur.execute("SELECT count(*), count(DISTINCT doc_type) FROM policy_chunks")
            n, types = cur.fetchone()
    print(f"ingested {n} chunks ({types} doc types) into policy_chunks @ {DATABASE_URL.split('@')[-1]}")


if __name__ == "__main__":
    main()
