"""Shared rules engine — the single evaluator behind labeling, runtime
deterministic checks, and policy corpus generation.

Design note: this module is deliberately dependency-light (yaml + stdlib)
so it can be imported by training scripts, the FastAPI app, and CI alike.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

RULES_PATH = Path(__file__).resolve().parents[2] / "rules" / "rules.yaml"

OPS = {
    "<=": lambda v, t: v <= t,
    ">=": lambda v, t: v >= t,
    "<": lambda v, t: v < t,
    ">": lambda v, t: v > t,
}

AUTHORITY_ORDER = ["underwriter", "credit_manager", "credit_committee"]


@dataclass
class RuleException:
    """A single policy exception raised by a deterministic rule check."""

    exception_code: str
    rule_id: str
    section: str
    title: str
    severity: str
    waiver_authority: str
    observed: float
    threshold: float
    field_name: str
    override: str | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    exceptions: list[RuleException] = field(default_factory=list)

    @property
    def exception_count(self) -> int:
        return len(self.exceptions)

    @property
    def max_severity(self) -> str:
        order = ["none", "low", "medium", "high", "severe"]
        worst = "none"
        for e in self.exceptions:
            if order.index(e.severity) > order.index(worst):
                worst = e.severity
        return worst

    @property
    def has_compliance_override(self) -> bool:
        return any(e.override == "compliance" for e in self.exceptions)


def load_rules(path: Path | None = None) -> dict:
    with open(path or RULES_PATH) as f:
        return yaml.safe_load(f)


def _rule_in_effect(rule: dict, as_of: dt.date) -> bool:
    eff = rule.get("effective")
    exp = rule.get("expires")
    if eff and as_of < eff:
        return False
    if exp and as_of >= exp:
        return False
    return True


def _rule_applies(rule: dict, record: dict) -> bool:
    applies = rule.get("applies_to", {})
    loan_type = applies.get("loan_type", "any")
    return loan_type in ("any", record.get("loan_type"))


def evaluate(record: dict, rules: dict | None = None,
             as_of: dt.date | None = None) -> EvaluationResult:
    """Evaluate a loan record against every applicable, in-effect rule.

    `record` must contain the feature fields referenced by rule checks
    (ltv_ratio, dti_ratio, fico_score, doc_completeness,
    income_discrepancy_pct, loan_amount).
    """
    rules = rules or load_rules()
    as_of = as_of or _parse_date(record.get("application_date")) or dt.date.today()
    result = EvaluationResult()

    for rule in rules["rules"]:
        if not _rule_in_effect(rule, as_of):
            continue
        if not _rule_applies(rule, record):
            continue
        check = rule["check"]
        value = record.get(check["field"])
        if value is None:
            continue
        if not OPS[check["op"]](value, check["threshold"]):
            result.exceptions.append(RuleException(
                exception_code=rule["exception_code"],
                rule_id=rule["id"],
                section=rule["section"],
                title=rule["title"],
                severity=rule["severity"],
                waiver_authority=rule["waiver_authority"],
                observed=float(value),
                threshold=float(check["threshold"]),
                field_name=check["field"],
                override=rule.get("override"),
                message=(
                    f"{rule['title']}: observed {value} vs "
                    f"{check['op']} {check['threshold']} (policy section {rule['section']})"
                ),
            ))
    return result


def compliance_risk_label(record: dict, rules: dict | None = None,
                          noise: float = 0.0) -> float:
    """Rule-derived continuous risk label in [0, 1] used to train the
    compliance scorer. Severity weights come from rules.yaml so the model's
    notion of risk shares an origin with the deterministic checks."""
    rules = rules or load_rules()
    weights = rules["severity_weights"]
    result = evaluate(record, rules)
    base = sum(weights[e.severity] for e in result.exceptions)
    # Mild continuous pressure from near-threshold values so the model
    # learns gradients, not just step functions.
    base += max(0.0, record.get("ltv_ratio", 0) - 0.70) * 0.4
    base += max(0.0, record.get("dti_ratio", 0) - 0.38) * 0.5
    base += max(0.0, (680 - record.get("fico_score", 760)) / 400)
    base += record.get("prior_exceptions_count", 0) * 0.03
    return float(min(1.0, max(0.0, base + noise)))


def needs_review_label(record: dict, risk: float,
                       result: EvaluationResult) -> int:
    """Rule-derived binary label for the escalation classifier."""
    if result.has_compliance_override:
        return 1
    if result.max_severity in ("high", "severe"):
        return 1
    if result.exception_count >= 2 and risk >= 0.45:
        return 1
    if record.get("loan_amount", 0) > 1_000_000 and risk >= 0.40:
        return 1
    return 0


def required_waiver_level(exception: RuleException) -> str:
    return exception.waiver_authority


def can_waive(exception: RuleException, approver_level: str) -> bool:
    """Authority enforcement: approvals below the required level are invalid;
    not_waivable exceptions can never be waived."""
    if exception.waiver_authority == "not_waivable":
        return False
    return (AUTHORITY_ORDER.index(approver_level)
            >= AUTHORITY_ORDER.index(exception.waiver_authority))


def _parse_date(value) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))
