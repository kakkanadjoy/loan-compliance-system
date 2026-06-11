"""LLM memo drafter tests — no key, no network: these verify the
graceful-degradation contract, which is the part CI can and should test.
The live LLM path is verified manually (it requires real credentials).
"""
import pytest

pytest.importorskip("openai")

from agent import llm_memo
from agent.review_agent import build_graph
from rag import generate_corpus
from rag.explain import FileCorpusStore

RECORD = {
    "loan_id": "T-LLM", "loan_type": "equipment",
    "loan_amount": 620_000.0, "ltv_ratio": 0.82, "dti_ratio": 0.40,
    "fico_score": 700, "income_discrepancy_pct": 0.02,
    "doc_completeness": 1.0, "application_date": "2026-05-12",
    "state": "IA", "employment_type": "established_business",
    "prior_exceptions_count": 0, "prior_escalation_rate": 0.05,
    "days_in_pipeline": 4,
}


def test_unconfigured_env_reports_unavailable(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    assert llm_memo.llm_available() is False


def test_drafter_falls_back_to_template_without_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    generate_corpus.generate(out_dir=tmp_path)
    graph = build_graph(
        store=FileCorpusStore(tmp_path), memo_drafter=llm_memo.llm_memo_drafter
    )
    final = graph.invoke({"record": RECORD, "path": []})
    # full workflow completed and produced the deterministic memo
    assert final["decision"] == "review_required"
    assert "REVIEW MEMO" in final["memo"]
    assert "E-101" in final["memo"]


def test_bad_credentials_degrade_not_crash(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://invalid.invalid")
    generate_corpus.generate(out_dir=tmp_path)
    graph = build_graph(
        store=FileCorpusStore(tmp_path), memo_drafter=llm_memo.llm_memo_drafter
    )
    final = graph.invoke({"record": RECORD, "path": []})
    # the call fails, the workflow survives, the facts still ship
    assert "E-101" in final["memo"]
    assert "template used" in final["memo"]


def test_project_endpoint_normalized(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://res.services.ai.azure.com/api/projects/loan-compliance",
    )
    cfg = llm_memo._config()
    assert cfg["endpoint"] == "https://res.services.ai.azure.com"
