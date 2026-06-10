"""Synthetic loan portfolio generator — ground truth first.

Generates loan records as the single seed of truth. Documents are rendered
FROM these records (render_documents.py); features are extracted back from
the documents at intake; CI's round-trip test asserts they match.

Includes:
- realistic correlated fields (FICO <-> rates of delinquency, LTV by type)
- ~10% deliberate cross-document discrepancies (stated vs documented income)
- four seeded demo loans (APP-2024-0712 / 0533 / 0847 / 0291) matching the
  presentation walkthrough exactly
- rule-derived labels via the shared rules engine
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app import rules_engine  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "loans"

LOAN_TYPES = ["equipment", "working_capital", "cre"]
STATES = ["IA", "IL", "MN", "MO", "NE", "TX", "CA", "FL"]
STATE_RISK = {"IA": 0, "IL": 1, "MN": 0, "MO": 1, "NE": 0, "TX": 1, "CA": 2, "FL": 2}
EMPLOYMENT = ["established_business", "newer_business", "seasonal"]


def _gen_record(rng: np.random.Generator, idx: int, rules: dict) -> dict:
    loan_type = LOAN_TYPES[rng.integers(0, len(LOAN_TYPES))]
    state = STATES[rng.integers(0, len(STATES))]

    fico = int(np.clip(rng.normal(715, 48), 560, 850))
    # Healthier credit correlates with healthier ratios.
    quality = (fico - 560) / 290.0
    ltv = float(np.clip(rng.normal(0.78 - 0.12 * quality, 0.08), 0.35, 0.98))
    dti = float(np.clip(rng.normal(0.44 - 0.10 * quality, 0.07), 0.15, 0.70))

    amount = float(np.round(rng.lognormal(mean=12.6, sigma=0.55), -3))
    amount = float(np.clip(amount, 50_000, 4_000_000))
    collateral_value = float(np.round(amount / ltv, -2))

    documented_income = float(np.round(rng.lognormal(13.2, 0.5), -2))
    # ~10% of loans get an injected stated-vs-documented discrepancy.
    if rng.random() < 0.10:
        bump = rng.uniform(0.12, 0.45)
        stated_income = float(np.round(documented_income * (1 + bump), -2))
    else:
        stated_income = float(np.round(documented_income * rng.uniform(0.98, 1.06), -2))
    income_discrepancy = abs(stated_income - documented_income) / documented_income

    required = rules["required_documents"][loan_type]
    missing: list[str] = []
    if rng.random() < 0.22:
        n_missing = 1 if rng.random() < 0.85 else 2
        candidates = [d for d in required if d != "application"]
        missing = list(rng.choice(candidates, size=n_missing, replace=False))
    doc_completeness = round((len(required) - len(missing)) / len(required), 2)

    app_date = dt.date(2026, 1, 1) + dt.timedelta(days=int(rng.integers(0, 150)))

    record = {
        "loan_id": f"APP-2026-{2000 + idx:04d}",
        "application_date": app_date.isoformat(),
        "loan_type": loan_type,
        "state": state,
        "state_risk_level": STATE_RISK[state],
        "employment_type": EMPLOYMENT[rng.integers(0, len(EMPLOYMENT))],
        "loan_amount": amount,
        "collateral_value": collateral_value,
        "ltv_ratio": round(amount / collateral_value, 2),
        "dti_ratio": round(dti, 2),
        "fico_score": fico,
        "stated_income": stated_income,
        "documented_income": documented_income,
        "income_discrepancy_pct": round(income_discrepancy, 3),
        "required_documents": required,
        "missing_documents": missing,
        "doc_completeness": doc_completeness,
        "is_jumbo": amount > 1_000_000,
        # System-derived features (live in DB, never in documents).
        "prior_exceptions_count": int(rng.poisson(0.6)),
        "prior_escalation_rate": round(float(rng.beta(1.2, 8.0)), 3),
        "days_in_pipeline": int(rng.integers(1, 35)),
    }
    return record


def _seeded_demo_loans(rules: dict) -> list[dict]:
    """The four walkthrough loans, exact values, deterministic."""
    base = {
        "application_date": "2026-05-12",
        "state": "IA",
        "state_risk_level": 0,
        "employment_type": "established_business",
        "prior_escalation_rate": 0.05,
        "days_in_pipeline": 4,
    }
    eq_docs = rules["required_documents"]["equipment"]
    wc_docs = rules["required_documents"]["working_capital"]
    demos = [
        # Clean fast-track loan.
        dict(base, loan_id="APP-2024-0712", loan_type="working_capital",
             loan_amount=180_000.0, collateral_value=264_700.0, ltv_ratio=0.68,
             dti_ratio=0.31, fico_score=745, stated_income=410_000.0,
             documented_income=405_000.0, income_discrepancy_pct=0.012,
             required_documents=wc_docs, missing_documents=[],
             doc_completeness=1.0, is_jumbo=False, prior_exceptions_count=0),
        # One minor doc exception -> manager review.
        dict(base, loan_id="APP-2024-0533", loan_type="working_capital",
             loan_amount=240_000.0, collateral_value=342_900.0, ltv_ratio=0.70,
             dti_ratio=0.39, fico_score=702, stated_income=380_000.0,
             documented_income=372_000.0, income_discrepancy_pct=0.022,
             required_documents=wc_docs, missing_documents=["financials"],
             doc_completeness=0.75, is_jumbo=False, prior_exceptions_count=0),
        # The walkthrough loan: LTV exception + missing insurance binder.
        dict(base, loan_id="APP-2024-0847", loan_type="equipment",
             loan_amount=620_000.0, collateral_value=756_000.0, ltv_ratio=0.82,
             dti_ratio=0.44, fico_score=689, stated_income=520_000.0,
             documented_income=505_000.0, income_discrepancy_pct=0.03,
             required_documents=eq_docs, missing_documents=["insurance_binder"],
             doc_completeness=0.83, is_jumbo=False, prior_exceptions_count=1),
        # Severe: income misrepresentation -> compliance override.
        dict(base, loan_id="APP-2024-0291", loan_type="equipment",
             loan_amount=1_250_000.0, collateral_value=1_513_000.0,
             ltv_ratio=0.83, dti_ratio=0.49, fico_score=664,
             stated_income=690_000.0, documented_income=515_000.0,
             income_discrepancy_pct=0.34, required_documents=eq_docs,
             missing_documents=[], doc_completeness=1.0, is_jumbo=True,
             prior_exceptions_count=2),
    ]
    return demos


def generate(n: int = 500, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    rules = rules_engine.load_rules()
    records = [_gen_record(rng, i, rules) for i in range(n)]
    records.extend(_seeded_demo_loans(rules))

    label_rng = np.random.default_rng(seed + 1)
    for rec in records:
        result = rules_engine.evaluate(rec, rules)
        noise = float(label_rng.normal(0, 0.03))
        risk = rules_engine.compliance_risk_label(rec, rules, noise=noise)
        rec["label_compliance_risk"] = round(risk, 4)
        rec["label_needs_review"] = rules_engine.needs_review_label(rec, risk, result)
        rec["ground_truth_exceptions"] = [e.exception_code for e in result.exceptions]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = generate(args.n, args.seed)
    out = DATA_DIR / "records.jsonl"
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    n_disc = sum(1 for r in records if r["income_discrepancy_pct"] > 0.10)
    n_exc = sum(1 for r in records if r["ground_truth_exceptions"])
    print(f"wrote {len(records)} records -> {out}")
    print(f"  with >=1 exception: {n_exc} | income discrepancies >10%: {n_disc}")
    print(f"  needs_review rate: {np.mean([r['label_needs_review'] for r in records]):.2%}")


if __name__ == "__main__":
    main()
