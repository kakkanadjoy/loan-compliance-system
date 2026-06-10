"""Explain pipeline tests — DB-free: corpus is generated into tmp_path and
read via FileCorpusStore; records are constructed inline.
"""
from pathlib import Path

from rag import generate_corpus
from rag.explain import FileCorpusStore, explain_record


def _store(tmp_path: Path) -> FileCorpusStore:
    generate_corpus.generate(out_dir=tmp_path)
    return FileCorpusStore(corpus_dir=tmp_path)


BASE = {
    "application_date": "2026-05-12",
    "state": "IA",
    "employment_type": "established_business",
    "prior_exceptions_count": 0,
    "prior_escalation_rate": 0.05,
    "days_in_pipeline": 4,
}


def test_clean_loan_fast_tracks(tmp_path):
    record = dict(
        BASE, loan_id="T-CLEAN", loan_type="working_capital",
        loan_amount=180_000.0, ltv_ratio=0.68, dti_ratio=0.31,
        fico_score=745, income_discrepancy_pct=0.01, doc_completeness=1.0,
    )
    exp = explain_record(record, store=_store(tmp_path))
    assert exp["exception_count"] == 0
    assert exp["routing"] == "fast_track"
    assert exp["max_severity"] == "none"


def test_ltv_breach_cites_policy_and_waiver_chain(tmp_path):
    record = dict(
        BASE, loan_id="T-LTV", loan_type="equipment",
        loan_amount=620_000.0, ltv_ratio=0.82, dti_ratio=0.40,
        fico_score=700, income_discrepancy_pct=0.02, doc_completeness=1.0,
    )
    exp = explain_record(record, store=_store(tmp_path))
    codes = [f["exception_code"] for f in exp["findings"]]
    assert "E-101" in codes
    ltv = next(f for f in exp["findings"] if f["exception_code"] == "E-101")
    assert ltv["policy_section"] == "4.2.1"
    assert "80%" in ltv["threshold"]
    assert ltv["waivable_by"]  # waivable, by at least one level
    assert "loan-to-value" in ltv["policy_text"].lower()
    assert exp["routing"] != "fast_track"


def test_misrepresentation_not_waivable_routes_to_compliance(tmp_path):
    record = dict(
        BASE, loan_id="T-FRAUD", loan_type="equipment",
        loan_amount=1_250_000.0, ltv_ratio=0.83, dti_ratio=0.49,
        fico_score=664, income_discrepancy_pct=0.34, doc_completeness=1.0,
    )
    exp = explain_record(record, store=_store(tmp_path))
    assert exp["compliance_override"] is True
    assert exp["routing"] == "compliance_review"
    assert exp["max_severity"] == "severe"
    fraud = next(f for f in exp["findings"] if f["exception_code"] == "E-141")
    assert fraud["waivable_by"] == []          # nobody can waive it
    assert fraud["routes_to"] == "compliance"
    assert "misrepresentation" in fraud["policy_text"].lower()
