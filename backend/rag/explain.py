"""Explain pipeline — the joining piece between the rules engine and the
policy corpus.

Input: a loan record (or loan_id looked up in data/loans/records.jsonl).
Output: a structured, audit-ready explanation:
  - every rule exception fired, with the loan's observed value vs threshold
  - the governing policy text, cited DETERMINISTICALLY by exception code
    (audit citations must be exact — hybrid search stays for free-text Q&A)
  - severity, waiver authority, and which approval levels could waive it
  - rule-derived risk score and a routing recommendation per rules.yaml

Two interchangeable policy stores:
  - FileCorpusStore: reads data/corpus/*.md directly (offline, CI-friendly)
  - DbPolicyStore:   reads the policy_chunks table in postgres

CLI:
    python rag/explain.py APP-2024-0847
    python rag/explain.py APP-2024-0291 --store file --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app import rules_engine  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "loans"
CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://compliance:compliance@localhost:5432/compliance"
)


# --------------------------------------------------------------------------
# Policy stores — exact lookup by exception code
# --------------------------------------------------------------------------

def _parse_frontmatter_doc(text: str) -> dict:
    _, fm, body = text.split("---", 2)
    meta = {}
    for line in fm.strip().splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    meta["body"] = body.strip()
    return meta


class FileCorpusStore:
    """Reads policy docs straight from the generated markdown corpus."""

    def __init__(self, corpus_dir: Path | None = None):
        self.corpus_dir = corpus_dir or CORPUS_DIR
        self._by_code: dict[str, dict] = {}
        for path in sorted(self.corpus_dir.glob("rule_*.md")):
            doc = _parse_frontmatter_doc(path.read_text(encoding="utf-8"))
            code = doc.get("exception_code")
            if code:
                self._by_code[code] = doc

    def by_exception_code(self, code: str) -> dict | None:
        return self._by_code.get(code)


class DbPolicyStore:
    """Reads policy docs from the policy_chunks table (requires the
    docker compose stack and a completed rag/ingest.py run)."""

    def __init__(self, database_url: str = DATABASE_URL):
        import psycopg  # local import: file mode shouldn't require psycopg

        self._conn = psycopg.connect(database_url)

    def by_exception_code(self, code: str) -> dict | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT exception_code, rule_id, section, title, severity,
                       waiver_authority, body
                FROM policy_chunks WHERE exception_code = %s
                """,
                (code,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        keys = ["exception_code", "rule_id", "section", "title", "severity",
                "waiver_authority", "body"]
        return dict(zip(keys, row))


def open_store(kind: str = "auto"):
    """auto: prefer the database, fall back to corpus files."""
    if kind == "file":
        return FileCorpusStore(), "file"
    if kind == "db":
        return DbPolicyStore(), "db"
    try:
        return DbPolicyStore(), "db"
    except Exception:
        return FileCorpusStore(), "file"


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------

def _fmt_value(field: str, value: float) -> str:
    if field in ("ltv_ratio", "dti_ratio", "doc_completeness",
                 "income_discrepancy_pct"):
        return f"{value:.0%}"
    if field == "loan_amount":
        return f"${value:,.0f}"
    return f"{value:g}"


def _routing(record: dict, result, risk: float, rules: dict) -> str:
    routing = rules["routing"]
    if result.has_compliance_override:
        return "compliance_review"
    needs_review = rules_engine.needs_review_label(record, risk, result)
    if risk >= routing["senior_band_at_or_above"] or needs_review:
        return "senior_review"
    if result.exception_count > 0:
        return "manager_review"
    if risk < routing["triage_fast_track_below"]:
        return "fast_track"
    return "manager_review"


def explain_record(record: dict, store=None, rules: dict | None = None) -> dict:
    """Produce the structured explanation for one loan record."""
    rules = rules or rules_engine.load_rules()
    store = store or FileCorpusStore()
    result = rules_engine.evaluate(record, rules)
    risk = rules_engine.compliance_risk_label(record, rules)

    findings = []
    for e in result.exceptions:
        policy = store.by_exception_code(e.exception_code) or {}
        waivable_by = [
            lvl for lvl in rules_engine.AUTHORITY_ORDER
            if rules_engine.can_waive(e, lvl)
        ]
        findings.append(
            {
                "exception_code": e.exception_code,
                "rule_id": e.rule_id,
                "policy_section": e.section,
                "title": e.title,
                "severity": e.severity,
                "observed": _fmt_value(e.field_name, e.observed),
                "threshold": _fmt_value(e.field_name, e.threshold),
                "field": e.field_name,
                "waiver_authority": e.waiver_authority,
                "waivable_by": waivable_by,  # empty list = not waivable
                "routes_to": e.override,     # e.g. "compliance", or None
                "policy_text": policy.get("body", "(policy text unavailable)"),
            }
        )

    return {
        "loan_id": record.get("loan_id", "(unknown)"),
        "loan_type": record.get("loan_type"),
        "loan_amount": record.get("loan_amount"),
        "exception_count": result.exception_count,
        "max_severity": result.max_severity,
        "compliance_override": result.has_compliance_override,
        "risk_score": round(risk, 4),
        "routing": _routing(record, result, risk, rules),
        "findings": findings,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def load_record(loan_id: str, records_path: Path | None = None) -> dict:
    path = records_path or (DATA_DIR / "records.jsonl")
    if not path.exists():
        sys.exit(f"{path} not found — run ml_training/synthetic_data.py first")
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("loan_id") == loan_id:
                return rec
    sys.exit(f"loan {loan_id} not found in {path}")


def format_explanation(exp: dict) -> str:
    lines = [
        f"Loan {exp['loan_id']}  ({exp['loan_type']}, "
        f"${exp['loan_amount']:,.0f})",
        f"Risk score: {exp['risk_score']}   "
        f"Max severity: {exp['max_severity']}   "
        f"Routing: {exp['routing'].upper()}",
        "",
    ]
    if not exp["findings"]:
        lines.append("No policy exceptions. Eligible per all applicable rules.")
        return "\n".join(lines)

    lines.append(f"{exp['exception_count']} policy exception(s):")
    for i, f_ in enumerate(exp["findings"], 1):
        waive = (
            "waivable by: " + ", ".join(f_["waivable_by"])
            if f_["waivable_by"]
            else "NOT WAIVABLE"
            + (f" — routes to {f_['routes_to']}" if f_["routes_to"] else "")
        )
        lines += [
            "",
            f"[{i}] {f_['exception_code']} — {f_['title']} "
            f"(section {f_['policy_section']}, severity {f_['severity']})",
            f"    observed {f_['observed']} vs allowed {f_['threshold']} "
            f"({f_['field']})",
            f"    {waive}",
            f"    policy: {' '.join(f_['policy_text'].split())[:220]}...",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("loan_id")
    parser.add_argument("--store", choices=["auto", "db", "file"], default="auto")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    record = load_record(args.loan_id)
    store, used = open_store(args.store)
    exp = explain_record(record, store)
    if args.as_json:
        print(json.dumps(exp, indent=2))
    else:
        print(format_explanation(exp))
        print(f"\n(policy store: {used})")


if __name__ == "__main__":
    main()
