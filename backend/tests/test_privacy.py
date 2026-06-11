"""PII redaction tests.

Testing philosophy: assert the sensitive VALUE is gone from the output —
never assert which label it got. Overlapping recognizers (an SSN's shape
resembles a phone number) can classify differently across environments;
what matters is removal. The PERSON test auto-skips where no spaCy model
is installed (e.g. a minimal CI) and runs everywhere else.
"""
import pytest

pytest.importorskip("presidio_analyzer")

from app import privacy  # noqa: E402

SAMPLE = (
    "Borrower John Smith (SSN 078-05-1120, john.smith@example.com, "
    "phone (515) 555-0142) card 4111 1111 1111 1111 stated income "
    "contradicts the tax documents."
)


def test_pattern_pii_values_removed():
    r = privacy.redact_text(SAMPLE)
    for value in ["078-05-1120", "john.smith@example.com",
                  "(515) 555-0142", "4111 1111 1111 1111"]:
        assert value not in r["redacted"], f"{value!r} leaked"
    # the analytical content survives
    assert "stated income" in r["redacted"]
    assert "tax documents" in r["redacted"]


def test_findings_report_contains_no_values():
    r = privacy.redact_text(SAMPLE)
    assert isinstance(r["findings"], dict)
    assert sum(r["findings"].values()) >= 4
    # keys are entity types only; nothing in the report echoes the input
    for key in r["findings"]:
        assert key.isupper()
        assert key not in SAMPLE


def test_person_name_redacted_when_model_available():
    if not privacy.nlp_available():
        pytest.skip("no spaCy model installed - person detection disabled")
    r = privacy.redact_text(SAMPLE)
    assert "John Smith" not in r["redacted"]
    assert r["findings"].get("PERSON", 0) >= 1


def test_redact_api_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.post("/redact", json={"text": SAMPLE})
    assert resp.status_code == 200
    body = resp.json()
    assert "078-05-1120" not in body["redacted"]
    assert "findings" in body
