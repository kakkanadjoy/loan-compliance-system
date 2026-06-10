# Loan Underwriting Compliance System

LangGraph-orchestrated compliance workflow for commercial loan underwriting.
Deterministic rules and two XGBoost models make every routing decision; a
single grounded LLM call (Claude via Bedrock) writes exception narratives.
Every loan terminates in a human review lane — the system prepares decisions,
it never makes them.

## Design principles
- **`rules/rules.yaml` is the single source of truth.** It generates the
  policy corpus for RAG, labels the training data, drives runtime
  deterministic checks, and defines waiver authority levels — so the models,
  the retrieved policy text, and the evals can never contradict each other.
- **Ground truth first.** Synthetic loan records seed everything; PDFs are
  rendered *from* records; intake extracts features *from* PDFs; the
  round-trip test in CI proves extraction recovers ground truth.
- **Honest naming.** This is a workflow, not an autonomous agent: one LLM
  node with one bounded degree of freedom (max 2 retrieval loops). Models can
  escalate severity; only rules can de-escalate; regulatory triggers cannot
  be de-escalated at all.

## Quick start (Phase 0/1)
```bash
make install     # python deps
make up          # postgres+pgvector, redis, mlflow (docker)
make data        # generate 500 records + render PDF packets
make train       # train both models -> MLflow + models/
make test        # round-trip + rules engine tests
```

MLflow UI: http://localhost:5000 (when using `make up`, set
`MLFLOW_TRACKING_URI=http://localhost:5000`).

## Repo map
```
rules/rules.yaml                      # source of truth: rules, severities, waiver authority
backend/app/rules_engine.py           # shared evaluator: labeling + runtime + corpus
backend/app/config.py                 # env-driven settings
backend/ml_training/synthetic_data.py # ground-truth records, seeded demo loans
backend/ml_training/render_documents.py # records -> PDF packets (10% discrepant)
backend/ml_training/extract.py        # packet -> features (early intake stage)
backend/ml_training/train_*.py        # XGBoost + SHAP + MLflow (both models)
backend/tests/test_round_trip.py      # CI centerpiece: render->extract==truth
```

## Phases
- [x] **0/1** Foundation, data architecture, both models
- [ ] **2** RAG: policy corpus generation, pgvector ingestion, hybrid retrieval
- [ ] **3** LangGraph pipeline + FastAPI + Bedrock + Presidio
- [ ] **4** React analyst console (queues, run trace, exception cards)
- [ ] **5** Golden-set evals in CI
- [ ] **6** Evidently drift, retrain + promotion gate
- [ ] **7** AWS deploy, demo recording
