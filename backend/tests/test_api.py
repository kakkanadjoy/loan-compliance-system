"""API tests — FastAPI TestClient with an injected FileCorpusStore,
so the full HTTP boundary is tested without postgres or docker.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_store
from rag import generate_corpus
from rag.explain import FileCorpusStore


@pytest.fixture()
def client(tmp_path):
    generate_corpus.generate(out_dir=tmp_path)
    app.dependency_overrides[get_store] = lambda: FileCorpusStore(tmp_path)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


CLEAN = {
    "loan_id": "T-CLEAN", "loan_type": "working_capital",
    "loan_amount": 180000, "ltv_ratio": 0.68, "dti_ratio": 0.31,
    "fico_score": 745, "income_discrepancy_pct": 0.01,
    "doc_completeness": 1.0, "application_date": "2026-05-12",
}

FRAUD = {
    "loan_id": "T-FRAUD", "loan_type": "equipment",
    "loan_amount": 1250000, "ltv_ratio": 0.83, "dti_ratio": 0.49,
    "fico_score": 664, "income_discrepancy_pct": 0.34,
    "doc_completeness": 1.0, "application_date": "2026-05-12",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_evaluate_clean_loan_fast_tracks(client):
    r = client.post("/evaluate", json=CLEAN)
    assert r.status_code == 200
    body = r.json()
    assert body["routing"] == "fast_track"
    assert body["exception_count"] == 0


def test_evaluate_fraud_routes_to_compliance_with_citation(client):
    r = client.post("/evaluate", json=FRAUD)
    assert r.status_code == 200
    body = r.json()
    assert body["routing"] == "compliance_review"
    fraud = next(f for f in body["findings"] if f["exception_code"] == "E-141")
    assert fraud["waivable_by"] == []
    assert "misrepresentation" in fraud["policy_text"].lower()


def test_invalid_record_rejected_422(client):
    bad = dict(CLEAN, fico_score=9000)  # impossible FICO
    r = client.post("/evaluate", json=bad)
    assert r.status_code == 422  # pydantic boundary, business logic never ran


def test_unknown_loan_404(client):
    r = client.get("/loans/APP-0000-XXXX/explain")
    assert r.status_code == 404
