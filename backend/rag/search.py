"""Hybrid policy retrieval — vector similarity + keyword search,
fused with Reciprocal Rank Fusion (RRF).

Why hybrid: vector search finds semantically similar policy text even when
no words overlap ("borrower exaggerated earnings" -> income misrepresentation
rule), but is weak on exact identifiers. Keyword search nails identifiers
("E-141", "section 6.2.2") but misses paraphrases. RRF combines both rank
lists without needing to calibrate their incomparable scores: each document
scores sum(1 / (60 + rank)) across the lists it appears in.

CLI:
    python rag/search.py "equipment loan above 80 percent LTV"
    python rag/search.py "E-141" --k 3
"""
from __future__ import annotations

import argparse
import os

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://compliance:compliance@localhost:5432/compliance"
)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard damping constant; rank 0 contributes 1/60, rank 9 contributes 1/69

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def search(query: str, k: int = 5, pool: int = 10) -> list[dict]:
    """Return top-k policy chunks for the query via hybrid retrieval.

    pool: how many candidates each retriever contributes before fusion.
    """
    emb = _get_model().encode([query], normalize_embeddings=True)[0]

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id FROM policy_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (emb, pool),
            )
            vector_ranked = [r[0] for r in cur.fetchall()]

            cur.execute(
                """
                SELECT doc_id FROM policy_chunks
                WHERE tsv @@ plainto_tsquery('english', %s)
                ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC
                LIMIT %s
                """,
                (query, query, pool),
            )
            keyword_ranked = [r[0] for r in cur.fetchall()]

            scores: dict[str, float] = {}
            for ranked in (vector_ranked, keyword_ranked):
                for rank, doc_id in enumerate(ranked):
                    scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

            top = sorted(scores, key=scores.get, reverse=True)[:k]
            if not top:
                return []

            cur.execute(
                """
                SELECT doc_id, doc_type, rule_id, section, title,
                       exception_code, severity, waiver_authority, body
                FROM policy_chunks WHERE doc_id = ANY(%s)
                """,
                (top,),
            )
            rows = {r[0]: r for r in cur.fetchall()}

    results = []
    for doc_id in top:
        r = rows[doc_id]
        results.append(
            {
                "doc_id": r[0], "doc_type": r[1], "rule_id": r[2],
                "section": r[3], "title": r[4], "exception_code": r[5],
                "severity": r[6], "waiver_authority": r[7], "body": r[8],
                "rrf_score": round(scores[doc_id], 4),
                "in_vector": doc_id in vector_ranked,
                "in_keyword": doc_id in keyword_ranked,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, k=args.k)
    if not results:
        print("no matches")
        return
    for i, r in enumerate(results, 1):
        via = "+".join(
            s for s, hit in (("vector", r["in_vector"]), ("keyword", r["in_keyword"])) if hit
        )
        head = f"[{i}] {r['title']}"
        if r["section"]:
            head += f"  (section {r['section']}"
            if r["exception_code"]:
                head += f", {r['exception_code']}"
            head += ")"
        print(head)
        print(f"    rrf={r['rrf_score']}  via {via}")
        snippet = " ".join(r["body"].split())[:160]
        print(f"    {snippet}...")
        print()


if __name__ == "__main__":
    main()
