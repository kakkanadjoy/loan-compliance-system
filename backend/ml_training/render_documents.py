"""Render loan records into PDF document packets.

Documents are generated FROM ground-truth records so that intake extraction
has verifiable truth (the round-trip test). Templates use stable
"Label: value" lines that the extractor parses; layouts are intentionally
simple but realistic enough to demo.

Discrepancies are honored at render time: the application form shows
stated_income while the tax summary shows documented_income — when these
differ beyond tolerance, the cross-document consistency check at intake
should catch it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

sys.path.append(str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "loans"


def _page(c: canvas.Canvas, title: str, lines: list[str]) -> None:
    width, height = LETTER
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, height - 1 * inch, title)
    c.setFont("Helvetica", 8)
    c.drawString(1 * inch, height - 1.2 * inch,
                 "Synthetic document generated for demonstration purposes only.")
    c.setFont("Helvetica", 11)
    y = height - 1.7 * inch
    for line in lines:
        c.drawString(1 * inch, y, line)
        y -= 0.28 * inch
    c.showPage()


def _money(v: float) -> str:
    return f"${v:,.0f}"


def render_packet(rec: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    missing = set(rec["missing_documents"])

    required = set(rec["required_documents"])

    def doc(name: str, title: str, lines: list[str]) -> None:
        if name in missing or name not in required:
            return
        path = out_dir / f"{name}.pdf"
        c = canvas.Canvas(str(path), pagesize=LETTER)
        _page(c, title, lines)
        c.save()
        written.append(path)

    doc("application", "Commercial Loan Application", [
        f"Loan ID: {rec['loan_id']}",
        f"Application Date: {rec['application_date']}",
        f"Loan Type: {rec['loan_type']}",
        f"State: {rec['state']}",
        f"Requested Amount: {_money(rec['loan_amount'])}",
        f"Stated Annual Income: {_money(rec['stated_income'])}",
        f"Employment Type: {rec['employment_type']}",
    ])

    doc("financials", "Business Financial Summary", [
        f"Loan ID: {rec['loan_id']}",
        f"Annual Revenue: {_money(rec['documented_income'] * 4.1)}",
        f"Net Operating Income: {_money(rec['documented_income'] * 1.4)}",
        f"Total Debt Obligations: {_money(rec['documented_income'] * rec['dti_ratio'])}",
        f"Debt-to-Income Ratio: {rec['dti_ratio']:.2f}",
    ])

    doc("credit_pull", "Consumer Credit Report — Principal Obligor", [
        f"Loan ID: {rec['loan_id']}",
        f"FICO Score: {rec['fico_score']}",
        f"Prior Exceptions On File: {rec['prior_exceptions_count']}",
        "Payment History: see attached tradelines (synthetic)",
    ])

    doc("tax_returns", "Tax Return Summary (3-Year)", [
        f"Loan ID: {rec['loan_id']}",
        f"Documented Annual Income: {_money(rec['documented_income'])}",
        "Returns Reviewed: 2023, 2024, 2025",
    ])

    doc("appraisal", "Collateral Appraisal Report", [
        f"Loan ID: {rec['loan_id']}",
        f"Appraised Value: {_money(rec['collateral_value'])}",
        f"Effective Appraisal Date: {rec['application_date']}",
        "Appraiser: Approved Panel (synthetic)",
    ])

    doc("purchase_agreement", "Purchase Agreement Summary", [
        f"Loan ID: {rec['loan_id']}",
        f"Purchase Price: {_money(rec['collateral_value'])}",
        f"Agreement Date: {rec['application_date']}",
    ])

    doc("insurance_binder", "Evidence of Insurance", [
        f"Loan ID: {rec['loan_id']}",
        f"Insured Collateral Value: {_money(rec['collateral_value'])}",
        "Coverage: All-risk, lender as loss payee (synthetic)",
    ])

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(DATA_DIR / "records.jsonl"))
    parser.add_argument("--limit", type=int, default=0,
                        help="render only the first N records (0 = all)")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(args.records)]
    if args.limit:
        # Always include the seeded demo loans regardless of limit.
        demos = [r for r in records if r["loan_id"].startswith("APP-2024-")]
        records = records[: args.limit] + demos

    count = 0
    for rec in records:
        render_packet(rec, DATA_DIR / "packets" / rec["loan_id"])
        count += 1
    print(f"rendered {count} packets -> {DATA_DIR / 'packets'}")


if __name__ == "__main__":
    main()
