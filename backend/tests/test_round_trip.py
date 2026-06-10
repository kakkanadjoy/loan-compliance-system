"""Round-trip test: records -> rendered PDFs -> extracted features == records.

This is the CI centerpiece. If document templates, the extractor, or the
generator drift apart, this fails — which is exactly the guarantee that
makes the demo's numbers trustworthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app import rules_engine  # noqa: E402
from ml_training import synthetic_data, render_documents, extract  # noqa: E402

TOL_MONEY = 1.0          # exact rendering, allow $1 parse slack
TOL_RATIO = 0.011        # ratios rendered at 2dp


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    out = tmp_path_factory.mktemp("packets")
    records = synthetic_data.generate(n=20, seed=7)
    for rec in records:
        render_documents.render_packet(rec, out / rec["loan_id"])
    return records, out


def test_round_trip_doc_derived_features(sample):
    records, out = sample
    for rec in records:
        got = extract.extract_packet(out / rec["loan_id"], rec["required_documents"])
        assert got["loan_id"] == rec["loan_id"]
        assert sorted(got["missing_documents"]) == sorted(rec["missing_documents"])
        assert abs(got["doc_completeness"] - rec["doc_completeness"]) < 0.01
        if "application" not in rec["missing_documents"]:
            assert abs(got["loan_amount"] - rec["loan_amount"]) <= TOL_MONEY
        if "credit_pull" not in rec["missing_documents"]:
            assert got["fico_score"] == rec["fico_score"]
        if "appraisal" not in rec["missing_documents"] and got.get("ltv_ratio"):
            assert abs(got["ltv_ratio"] - rec["ltv_ratio"]) <= TOL_RATIO


def test_demo_loan_0847_exact(sample):
    records, out = sample
    rec = next(r for r in records if r["loan_id"] == "APP-2024-0847")
    got = extract.extract_packet(out / rec["loan_id"], rec["required_documents"])
    assert got["fico_score"] == 689
    assert got["ltv_ratio"] == 0.82
    assert got["missing_documents"] == ["insurance_binder"]
    assert got["doc_completeness"] == 0.83


def test_rules_engine_demo_expectations():
    """Golden expectations for the four walkthrough loans."""
    rules = rules_engine.load_rules()
    records = {r["loan_id"]: r for r in synthetic_data.generate(n=0, seed=1)}

    clean = rules_engine.evaluate(records["APP-2024-0712"], rules)
    assert clean.exception_count == 0

    walkthrough = rules_engine.evaluate(records["APP-2024-0847"], rules)
    codes = {e.exception_code for e in walkthrough.exceptions}
    assert codes == {"E-101", "E-130"}          # LTV + missing binder
    assert walkthrough.max_severity == "medium"
    assert not walkthrough.has_compliance_override

    severe = rules_engine.evaluate(records["APP-2024-0291"], rules)
    severe_codes = {e.exception_code for e in severe.exceptions}
    assert "E-141" in severe_codes               # misrepresentation
    assert severe.has_compliance_override        # hard override fires


def test_waiver_authority_enforcement():
    rules = rules_engine.load_rules()
    rec = next(r for r in synthetic_data.generate(n=0, seed=1)
               if r["loan_id"] == "APP-2024-0847")
    result = rules_engine.evaluate(rec, rules)
    ltv = next(e for e in result.exceptions if e.exception_code == "E-101")
    assert not rules_engine.can_waive(ltv, "underwriter")
    assert rules_engine.can_waive(ltv, "credit_manager")

    rec291 = next(r for r in synthetic_data.generate(n=0, seed=1)
                  if r["loan_id"] == "APP-2024-0291")
    sev = next(e for e in rules_engine.evaluate(rec291, rules).exceptions
               if e.exception_code == "E-141")
    assert not rules_engine.can_waive(sev, "credit_committee")  # not_waivable
