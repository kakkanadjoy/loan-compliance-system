"""Extract loan features from a rendered PDF packet.

This is the early version of the runtime intake stage. Because templates are
ours, extraction uses labeled-line parsing; the round-trip test proves it
recovers ground truth. The same module gains Presidio PII tokenization and
LLM-assisted extraction for messier documents in Phase 3 — behind the same
interface, so the round-trip test keeps guarding it.
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

_MONEY = r"\$([\d,]+)"


def _read_text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _money(pattern: str, text: str) -> float | None:
    m = re.search(pattern + r"\s*" + _MONEY, text)
    return float(m.group(1).replace(",", "")) if m else None


def _line(pattern: str, text: str) -> str | None:
    m = re.search(pattern + r"\s*(.+)", text)
    return m.group(1).strip() if m else None


def extract_packet(packet_dir: Path, required_documents: list[str]) -> dict:
    """Parse every present PDF and assemble the extracted feature set,
    including doc completeness and the cross-document income check."""
    texts: dict[str, str] = {}
    for name in required_documents:
        pdf = packet_dir / f"{name}.pdf"
        if pdf.exists():
            texts[name] = _read_text(pdf)

    present = set(texts)
    missing = [d for d in required_documents if d not in present]
    out: dict = {
        "missing_documents": missing,
        "doc_completeness": round((len(required_documents) - len(missing))
                                  / len(required_documents), 2),
    }

    app = texts.get("application", "")
    out["loan_id"] = _line(r"Loan ID:", app)
    out["loan_type"] = _line(r"Loan Type:", app)
    out["application_date"] = _line(r"Application Date:", app)
    out["loan_amount"] = _money(r"Requested Amount:", app)
    out["stated_income"] = _money(r"Stated Annual Income:", app)

    credit = texts.get("credit_pull", "")
    fico = _line(r"FICO Score:", credit)
    out["fico_score"] = int(fico) if fico else None

    fin = texts.get("financials", "")
    dti = _line(r"Debt-to-Income Ratio:", fin)
    out["dti_ratio"] = float(dti) if dti else None

    appraisal = texts.get("appraisal", "")
    out["collateral_value"] = _money(r"Appraised Value:", appraisal)
    if out.get("loan_amount") and out.get("collateral_value"):
        out["ltv_ratio"] = round(out["loan_amount"] / out["collateral_value"], 2)

    tax = texts.get("tax_returns", "")
    out["documented_income"] = _money(r"Documented Annual Income:", tax)

    # Cross-document consistency: stated (application) vs documented (tax).
    if out.get("stated_income") and out.get("documented_income"):
        out["income_discrepancy_pct"] = round(
            abs(out["stated_income"] - out["documented_income"])
            / out["documented_income"], 3)

    return out
