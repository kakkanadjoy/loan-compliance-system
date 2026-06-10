"""Policy corpus generator — rules.yaml -> markdown documents.

Emits one markdown document per rule, plus governance documents
(waiver authority, required documents, routing thresholds), all derived
from the same rules.yaml that drives labeling and runtime checks.
The retrieved policy text can therefore never contradict the rules engine.

Each document carries YAML frontmatter with structured metadata; the
ingestion script (ingest.py) stores that metadata alongside embeddings
so retrieval can filter on it (e.g. only rules for loan_type=equipment).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app import rules_engine  # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

OP_PHRASE = {
    "<=": "must not exceed",
    "<": "must be below",
    ">=": "must be at least",
    ">": "must be above",
}

FIELD_PHRASE = {
    "ltv_ratio": "the loan-to-value (LTV) ratio",
    "dti_ratio": "the debt-to-income (DTI) ratio",
    "fico_score": "the principal obligor's credit score (FICO)",
    "doc_completeness": "documentation completeness",
    "income_discrepancy_pct": "the variance between stated and documented income",
    "loan_amount": "the loan amount",
}


def _fmt_threshold(field: str, value: float) -> str:
    if field in ("ltv_ratio", "dti_ratio", "doc_completeness", "income_discrepancy_pct"):
        return f"{value:.0%}"
    if field == "loan_amount":
        return f"${value:,.0f}"
    return f"{value:g}"


def _frontmatter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _rule_doc(rule: dict, manual: str) -> tuple[str, str]:
    """Returns (doc_id, markdown) for one rule."""
    check = rule["check"]
    field, op, threshold = check["field"], check["op"], check["threshold"]
    applies = rule.get("applies_to", {}).get("loan_type", "any")
    applies_phrase = (
        "all loan types" if applies == "any" else f"{applies.replace('_', ' ')} loans"
    )
    waiver = rule["waiver_authority"]
    if waiver == "not_waivable":
        waiver_phrase = (
            "This requirement is not waivable at any authority level."
            + (
                f" Findings route directly to the {rule['override']} officer."
                if rule.get("override")
                else ""
            )
        )
    else:
        waiver_phrase = (
            f"Exceptions may be approved at {waiver.replace('_', ' ')} level or above."
        )

    operational = (
        f"For {applies_phrase}, {FIELD_PHRASE.get(field, field)} "
        f"{OP_PHRASE[op]} {_fmt_threshold(field, threshold)}. "
        f"Violations raise exception code {rule['exception_code']} "
        f"with {rule['severity']} severity. {waiver_phrase}"
    )

    doc_id = f"rule_{rule['section'].replace('.', '-')}_{rule['id']}"
    fm = _frontmatter(
        {
            "doc_id": doc_id,
            "doc_type": "rule",
            "rule_id": rule["id"],
            "section": rule["section"],
            "title": rule["title"],
            "exception_code": rule["exception_code"],
            "severity": rule["severity"],
            "waiver_authority": waiver,
            "applies_to": applies,
            "effective": rule.get("effective", ""),
        }
    )
    body = "\n\n".join(
        [
            f"# {manual} — Section {rule['section']}: {rule['title']}",
            " ".join(rule["text"].split()),
            f"**Operational summary.** {operational}",
        ]
    )
    return doc_id, f"{fm}\n\n{body}\n"


def _governance_docs(rules: dict, manual: str) -> list[tuple[str, str]]:
    docs = []

    levels = rules["meta"]["authority_levels"]
    waiver_lines = [
        f"- **{r['id']}** ({r['section']}, {r['title']}): "
        + (
            "not waivable"
            + (f", routes to {r['override']}" if r.get("override") else "")
            if r["waiver_authority"] == "not_waivable"
            else r["waiver_authority"].replace("_", " ")
        )
        for r in rules["rules"]
    ]
    body = "\n\n".join(
        [
            f"# {manual} — Waiver authority hierarchy",
            "Authority levels in ascending order: "
            + " < ".join(levels)
            + ". A waiver may be granted by the listed level or any higher level. "
            "Rules marked not waivable cannot be waived at any level.",
            "Waiver authority by rule:",
            "\n".join(waiver_lines),
        ]
    )
    docs.append(
        (
            "governance_waiver_authority",
            _frontmatter(
                {"doc_id": "governance_waiver_authority", "doc_type": "governance",
                 "title": "Waiver authority hierarchy"}
            )
            + "\n\n" + body + "\n",
        )
    )

    req = rules["required_documents"]
    req_lines = [
        f"- **{lt.replace('_', ' ')} loans**: " + ", ".join(d.replace("_", " ") for d in docs_)
        for lt, docs_ in req.items()
    ]
    body = "\n\n".join(
        [
            f"# {manual} — Required documentation by loan type",
            "A complete application package must include every document listed "
            "for the loan type. Missing items raise exception E-130 per section 5.1.1.",
            "\n".join(req_lines),
        ]
    )
    docs.append(
        (
            "governance_required_documents",
            _frontmatter(
                {"doc_id": "governance_required_documents", "doc_type": "governance",
                 "title": "Required documentation by loan type"}
            )
            + "\n\n" + body + "\n",
        )
    )

    routing = rules["routing"]
    body = "\n\n".join(
        [
            f"# {manual} — Routing thresholds",
            textwrap.dedent(
                f"""\
                Applications with a compliance risk score below
                {routing['triage_fast_track_below']:.2f} are eligible for fast-track
                processing. Scores at or above {routing['senior_band_at_or_above']:.2f}
                route to the senior review band. An escalation probability at or above
                {routing['escalation_probability_at_or_above']:.2f} routes the
                application to a senior reviewer regardless of risk score. A maximum of
                {routing['max_resubmit_attempts']} resubmission attempts is permitted."""
            ).replace("\n", " "),
        ]
    )
    docs.append(
        (
            "governance_routing_thresholds",
            _frontmatter(
                {"doc_id": "governance_routing_thresholds", "doc_type": "governance",
                 "title": "Routing thresholds"}
            )
            + "\n\n" + body + "\n",
        )
    )
    return docs


def generate(out_dir: Path | None = None) -> list[Path]:
    out = out_dir or CORPUS_DIR
    out.mkdir(parents=True, exist_ok=True)
    rules = rules_engine.load_rules()
    manual = rules["meta"]["policy_manual"]

    written: list[Path] = []
    for rule in rules["rules"]:
        doc_id, md = _rule_doc(rule, manual)
        path = out / f"{doc_id}.md"
        path.write_text(md, encoding="utf-8")
        written.append(path)

    for doc_id, md in _governance_docs(rules, manual):
        path = out / f"{doc_id}.md"
        path.write_text(md, encoding="utf-8")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    written = generate(args.out)
    rule_docs = sum(1 for p in written if p.name.startswith("rule_"))
    gov_docs = len(written) - rule_docs
    print(f"wrote {len(written)} corpus documents -> {written[0].parent}")
    print(f"  rule documents: {rule_docs} | governance documents: {gov_docs}")


if __name__ == "__main__":
    main()
