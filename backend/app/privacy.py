"""PII redaction — the privacy gate for all free text.

Any text that will be stored, logged, embedded, or sent to an LLM should
pass through redact_text() first. PII is replaced with typed placeholders
(<PERSON>, <US_SSN>, ...); the findings report contains ONLY entity types
and counts — never the detected values, so the report itself can't leak.

Detection (Microsoft Presidio):
- pattern recognizers with validation: SSN, email, phone, credit card,
  bank account — these work everywhere, no language model needed
- NLP-based recognition (PERSON names) needs a spaCy model. Default:
  en_core_web_sm (small, ~12MB: `python -m spacy download en_core_web_sm`).
  If no model is installed, the module degrades gracefully: pattern
  entities still redact, person-name detection is disabled, and
  `nlp_available` in the result says which mode you're in.

Usage:
    from app.privacy import redact_text
    result = redact_text("John Smith, SSN 078-05-1120")
    result["redacted"]  -> "<PERSON>, SSN <US_SSN>"
    result["findings"]  -> {"PERSON": 1, "US_SSN": 1}
"""
from __future__ import annotations

import os
from collections import Counter

DEFAULT_ENTITIES = [
    "PERSON",
    "US_SSN",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_BANK_NUMBER",
]

SPACY_MODEL = os.environ.get("PII_SPACY_MODEL", "en_core_web_sm")

_analyzer = None
_anonymizer = None
_nlp_available: bool | None = None


def _build_engines():
    """Lazy singleton build — presidio + spaCy load is slow, do it once."""
    global _analyzer, _anonymizer, _nlp_available
    if _analyzer is not None:
        return

    import spacy
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    if spacy.util.is_package(SPACY_MODEL):
        conf = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=conf).create_engine()
        _nlp_available = True
    else:
        # No model installed: blank pipeline. Pattern recognizers still
        # work; NER-based entities (PERSON) will not be detected.
        spacy.blank("en").to_disk("/tmp/_blank_en") if False else None
        conf = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "xx_blank"}],
        }
        # NlpEngineProvider can't load a nonexistent package; build the
        # engine around an in-memory blank pipeline instead.
        from presidio_analyzer.nlp_engine import SpacyNlpEngine

        class _BlankSpacyEngine(SpacyNlpEngine):
            def __init__(self):
                super().__init__(models=[{"lang_code": "en",
                                          "model_name": "en_core_web_sm"}])

            def load(self):
                import spacy as _sp
                self.nlp = {"en": _sp.blank("en")}

        nlp_engine = _BlankSpacyEngine()
        nlp_engine.load()
        _nlp_available = False

    _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    _anonymizer = AnonymizerEngine()


def nlp_available() -> bool:
    _build_engines()
    return bool(_nlp_available)


def redact_text(text: str, entities: list[str] | None = None) -> dict:
    """Redact PII from text.

    Returns:
        {
          "redacted": str,            # text with <ENTITY_TYPE> placeholders
          "findings": {entity: count},# types and counts ONLY — no values
          "nlp_available": bool,      # False = person-name detection off
        }
    """
    _build_engines()
    entities = entities or DEFAULT_ENTITIES

    results = _analyzer.analyze(text=text, entities=entities, language="en")
    redacted = _anonymizer.anonymize(text=text, analyzer_results=results).text
    findings = Counter(r.entity_type for r in results)
    return {
        "redacted": redacted,
        "findings": dict(sorted(findings.items())),
        "nlp_available": bool(_nlp_available),
    }
