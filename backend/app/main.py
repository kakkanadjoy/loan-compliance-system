"""Loan Compliance Service — the HTTP boundary around everything built so far.

Endpoints:
    GET  /health                    liveness + which policy store is active
    POST /evaluate                  judge a submitted loan record (validated JSON in,
                                    audit-ready explanation out)
    GET  /loans/{loan_id}/explain   explanation for a known portfolio loan
    GET  /policy/search?q=&k=       hybrid retrieval over the policy manual

Run from backend/ with the venv active:
    uvicorn app.main:app --reload
Interactive docs: http://localhost:8000/docs

Design notes:
- Validation at the boundary: LoanRecord (pydantic) rejects malformed input
  with a 422 before any business logic runs.
- The policy store is opened once at startup: postgres if reachable,
  corpus files otherwise (same auto-fallback as rag/explain.py). Tests
  inject a FileCorpusStore via dependency override — no database in CI.
- Hybrid search imports sentence-transformers lazily so the service starts
  fast and runs even where torch isn't installed; the endpoint degrades to
  a 503 with a clear message instead of taking the whole app down.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.explain import explain_record, load_record, open_store  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    store, kind = open_store("auto")
    app.state.store = store
    app.state.store_kind = kind
    yield


app = FastAPI(
    title="Loan Compliance Service",
    version="0.3.0",
    description="Deterministic rules + policy citations + routing, over HTTP.",
    lifespan=lifespan,
)


def get_store(request: Request):
    return request.app.state.store


# --------------------------------------------------------------------------
# Request/response models — the validated boundary
# --------------------------------------------------------------------------

class LoanRecord(BaseModel):
    """The fields the rules engine needs; extra fields are accepted and
    passed through (system-derived features, etc.)."""

    model_config = ConfigDict(extra="allow")

    loan_id: str = "(unsubmitted)"
    loan_type: Literal["equipment", "working_capital", "cre"]
    loan_amount: float = Field(gt=0)
    ltv_ratio: float = Field(gt=0, le=1.5)
    dti_ratio: float = Field(ge=0, le=1.5)
    fico_score: int = Field(ge=300, le=850)
    income_discrepancy_pct: float = Field(ge=0, le=5)
    doc_completeness: float = Field(ge=0, le=1)
    application_date: str | None = None
    prior_exceptions_count: int = 0


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "policy_store": request.app.state.store_kind}


class RedactRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


@app.post("/redact")
def redact(req: RedactRequest) -> dict:
    """PII redaction gate: text in, placeholders out. The findings report
    contains entity types and counts only — never the detected values."""
    try:
        from app import privacy  # lazy: presidio + spaCy load is heavy
        return privacy.redact_text(req.text)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="redaction unavailable: presidio not installed",
        )


@app.post("/evaluate")
def evaluate(record: LoanRecord, store=Depends(get_store)) -> dict:
    """Judge a loan record: exceptions, citations, waiver chains, routing."""
    return explain_record(record.model_dump(), store=store)


@app.get("/loans/{loan_id}/explain")
def explain_loan(loan_id: str, store=Depends(get_store)) -> dict:
    """Explanation for a loan already in the portfolio (records.jsonl)."""
    try:
        record = load_record(loan_id)
    except SystemExit as e:  # load_record exits on missing file/loan
        raise HTTPException(status_code=404, detail=str(e))
    return explain_record(record, store=store)


@app.get("/loans")
def list_loans(limit: int = Query(default=18, ge=1, le=50)) -> list:
    """The review queue: demo loans first, then the synthetic portfolio."""
    import json as _json

    from rag import explain as explain_module

    path = explain_module.DATA_DIR / "records.jsonl"
    if not path.exists():
        return []
    demos, rest = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = _json.loads(line)
            row = {
                "loan_id": rec["loan_id"],
                "loan_type": rec["loan_type"],
                "loan_amount": rec["loan_amount"],
                "needs_review": bool(rec.get("label_needs_review")),
                "exceptions": len(rec.get("ground_truth_exceptions", [])),
            }
            (demos if rec["loan_id"].startswith("APP-2024") else rest).append(row)
    return (demos + rest)[:limit]


@app.get("/loans/{loan_id}/review")
def review_loan_endpoint(loan_id: str, store=Depends(get_store)) -> dict:
    """Run the LangGraph review agent: gather -> branch -> memo.
    Uses the LLM drafter when credentials are configured, the deterministic
    template otherwise — same facts either way."""
    from agent.llm_memo import llm_available, llm_memo_drafter
    from agent.review_agent import build_graph, template_memo

    try:
        record = load_record(loan_id)
    except SystemExit as e:
        raise HTTPException(status_code=404, detail=str(e))

    if llm_available():
        drafter, drafter_name = llm_memo_drafter, "azure llm"
    else:
        drafter, drafter_name = template_memo, "template"
    graph = build_graph(store=store, memo_drafter=drafter)
    final = graph.invoke({"record": record, "path": []})
    return {
        "loan_id": loan_id,
        "decision": final["decision"],
        "queue": final["queue"],
        "minimum_authority": final["minimum_authority"],
        "blocking_codes": final["blocking_codes"],
        "path": final["path"],
        "memo": final["memo"],
        "drafter": drafter_name,
    }


@app.get("/policy/search")
def policy_search(
    q: str = Query(min_length=2),
    k: int = Query(default=5, ge=1, le=10),
) -> dict:
    """Free-text hybrid retrieval over the policy manual."""
    try:
        from rag.search import search  # lazy: pulls sentence-transformers
        results = search(q, k=k)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="search unavailable: sentence-transformers not installed",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"search unavailable (is the docker stack up and ingested?): {e}",
        )
    return {"query": q, "results": results}
