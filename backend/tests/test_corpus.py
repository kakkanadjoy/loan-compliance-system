"""Corpus generation tests — every rule must appear in the corpus,
with its exception code, threshold language, and waiver authority intact.
Runs without any database; pure rules.yaml -> markdown verification.
"""
from pathlib import Path

from app import rules_engine
from rag import generate_corpus


def test_corpus_covers_every_rule(tmp_path: Path):
    written = generate_corpus.generate(out_dir=tmp_path)
    rules = rules_engine.load_rules()

    assert len(written) == len(rules["rules"]) + 3  # rules + 3 governance docs

    all_text = "\n".join(p.read_text(encoding="utf-8") for p in written)
    for rule in rules["rules"]:
        assert rule["id"] in all_text, f"rule {rule['id']} missing from corpus"
        assert rule["exception_code"] in all_text
        assert f"Section {rule['section']}" in all_text


def test_rule_doc_frontmatter_roundtrip(tmp_path: Path):
    generate_corpus.generate(out_dir=tmp_path)
    rules = {r["id"]: r for r in rules_engine.load_rules()["rules"]}

    for path in tmp_path.glob("rule_*.md"):
        text = path.read_text(encoding="utf-8")
        fm = text.split("---")[1]
        meta = {}
        for line in fm.strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        rule = rules[meta["rule_id"]]
        assert meta["exception_code"] == rule["exception_code"]
        assert meta["severity"] == rule["severity"]
        assert meta["waiver_authority"] == rule["waiver_authority"]


def test_not_waivable_language_present(tmp_path: Path):
    generate_corpus.generate(out_dir=tmp_path)
    doc = (tmp_path / "rule_6-2-2_INC-ALL-002.md").read_text(encoding="utf-8")
    assert "not waivable" in doc.lower()
    assert "compliance" in doc.lower()
