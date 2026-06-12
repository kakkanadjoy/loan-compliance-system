"""Frontend + agent-endpoint tests — DB-free, records via tmp file."""
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_store
from rag import explain as explain_module
from rag import generate_corpus
from rag.explain import FileCorpusStore

DEMO = {
    "loan_id": "APP-2024-0291", "loan_type": "equipment",
    "loan_amount": 1_250_000.0, "ltv_ratio": 0.83, "dti_ratio": 0.49,
    "fico_score": 664, "income_discrepancy_pct": 0.34, "doc_completeness": 1.0,
    "application_date": "2026-05-12", "state": "IA",
    "employment_type": "established_business", "prior_exceptions_count": 2,
    "prior_escalation_rate": 0.05, "days_in_pipeline": 4,
    "label_needs_review": 1, "ground_truth_exceptions": ["E-101", "E-141"],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # records.jsonl in a tmp data dir, corpus in a tmp corpus dir
    data_dir = tmp_path / "loans"
    data_dir.mkdir()
    (data_dir / "records.jsonl").write_text(json.dumps(DEMO) + "\n", encoding="utf-8")
    monkeypatch.setattr(explain_module, "DATA_DIR", data_dir)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)

    corpus = tmp_path / "corpus"
    generate_corpus.generate(out_dir=corpus)
    app.dependency_overrides[get_store] = lambda: FileCorpusStore(corpus)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_root_serves_dashboard(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Loan Compliance Desk" in r.text


def test_queue_lists_demo_first(client):
    r = client.get("/loans")
    assert r.status_code == 200
    rows = r.json()
    assert rows[0]["loan_id"] == "APP-2024-0291"
    assert rows[0]["needs_review"] is True
    assert rows[0]["exceptions"] == 2


def test_review_endpoint_runs_agent_with_template_fallback(client):
    r = client.get("/loans/APP-2024-0291/review")
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "escalated_to_compliance"
    assert body["blocking_codes"] == ["E-141"]
    assert body["path"] == ["gather", "compliance_escalation", "draft_memo"]
    assert body["drafter"] == "template"
    assert "E-141" in body["memo"]


def test_review_unknown_loan_404(client):
    assert client.get("/loans/APP-0000-XXXX/review").status_code == 404
