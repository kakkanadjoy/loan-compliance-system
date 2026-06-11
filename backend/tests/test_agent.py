"""Review agent tests — DB-free (FileCorpusStore over a tmp corpus).
Asserts the agent's three behaviors: the path taken through the graph,
the decision reached, and the authority arithmetic.
"""
from pathlib import Path

from agent.review_agent import review_loan
from rag import generate_corpus
from rag.explain import FileCorpusStore


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


def test_clean_loan_auto_approves(tmp_path):
    record = dict(
        BASE, loan_id="T-CLEAN", loan_type="working_capital",
        loan_amount=180_000.0, ltv_ratio=0.68, dti_ratio=0.31,
        fico_score=745, income_discrepancy_pct=0.01, doc_completeness=1.0,
    )
    final = review_loan(record, store=_store(tmp_path))
    assert final["path"] == ["gather", "auto_approve", "draft_memo"]
    assert final["decision"] == "auto_approved"
    assert final["minimum_authority"] is None
    assert "fast-track" in final["memo"].lower()


def test_review_loan_computes_minimum_authority(tmp_path):
    record = dict(
        BASE, loan_id="T-LTV", loan_type="equipment",
        loan_amount=620_000.0, ltv_ratio=0.82, dti_ratio=0.40,
        fico_score=700, income_discrepancy_pct=0.02, doc_completeness=0.83,
    )
    final = review_loan(record, store=_store(tmp_path))
    assert final["path"] == ["gather", "waiver_analysis", "draft_memo"]
    assert final["decision"] == "review_required"
    # both findings are credit_manager-waivable -> minimum is credit_manager
    assert final["minimum_authority"] == "credit_manager"
    assert "E-101" in final["memo"]


def test_committee_level_finding_raises_minimum_authority(tmp_path):
    # income variance 15% fires E-140 (committee-waivable) but stays
    # under the 25% misrepresentation override
    record = dict(
        BASE, loan_id="T-VAR", loan_type="working_capital",
        loan_amount=300_000.0, ltv_ratio=0.70, dti_ratio=0.35,
        fico_score=720, income_discrepancy_pct=0.15, doc_completeness=1.0,
    )
    final = review_loan(record, store=_store(tmp_path))
    assert final["decision"] == "review_required"
    assert final["minimum_authority"] == "credit_committee"


def test_fraud_escalates_and_blocks(tmp_path):
    record = dict(
        BASE, loan_id="T-FRAUD", loan_type="equipment",
        loan_amount=1_250_000.0, ltv_ratio=0.83, dti_ratio=0.49,
        fico_score=664, income_discrepancy_pct=0.34, doc_completeness=1.0,
    )
    final = review_loan(record, store=_store(tmp_path))
    assert final["path"] == ["gather", "compliance_escalation", "draft_memo"]
    assert final["decision"] == "escalated_to_compliance"
    assert final["blocking_codes"] == ["E-141"]
    assert final["minimum_authority"] is None
    assert "no credit authority" in final["memo"].lower()
