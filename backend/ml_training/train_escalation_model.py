"""Train the escalation classifier (Option 3).

Predicts P(needs human review) from compliance-checker outputs plus loan
metadata. Recall is the metric that matters: a missed escalation is a risky
loan reviewed by the wrong level; a false positive is merely an extra review.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import (precision_score, recall_score,  # noqa: E402
                             roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402

from app import rules_engine  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "loans" / "records.jsonl"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

SEV_MAP = {"none": 0, "low": 1, "medium": 2, "high": 3, "severe": 4}

FEATURES: dict[str, str] = {
    "compliance_risk_score": "model:compliance_scorer",
    "exception_count": "rules_engine:deterministic",
    "max_exception_severity": "rules_engine:deterministic",
    "loan_amount_band": "doc:application",
    "is_jumbo": "doc:application",
    "doc_completeness": "doc:packet_inventory",
    "prior_escalation_rate": "system:review_history",
    "days_in_pipeline": "system:workflow",
}
TARGET = "label_needs_review"


def build_frame() -> pd.DataFrame:
    """Assemble escalation features the same way runtime will: rules engine
    outputs + the trained compliance model's score."""
    df = pd.DataFrame([json.loads(l) for l in open(DATA)])
    rules = rules_engine.load_rules()

    results = [rules_engine.evaluate(rec, rules) for rec in df.to_dict("records")]
    df["exception_count"] = [r.exception_count for r in results]
    df["max_exception_severity"] = [SEV_MAP[r.max_severity] for r in results]
    df["loan_amount_band"] = pd.cut(
        df["loan_amount"], [0, 250_000, 500_000, 1_000_000, np.inf],
        labels=[0, 1, 2, 3]).astype(int)
    df["is_jumbo"] = df["is_jumbo"].astype(int)

    comp_path = MODELS_DIR / "compliance_risk.json"
    if comp_path.exists():
        from ml_training.train_compliance_model import FEATURES as CF, LOAN_TYPE_MAP, EMP_MAP
        booster = xgb.XGBRegressor()
        booster.load_model(comp_path)
        feats = df.copy()
        feats["loan_type_enc"] = feats["loan_type"].map(LOAN_TYPE_MAP)
        feats["employment_type_enc"] = feats["employment_type"].map(EMP_MAP)
        df["compliance_risk_score"] = np.clip(
            booster.predict(feats[list(CF)]), 0, 1)
    else:  # fall back to label if scorer not trained yet
        df["compliance_risk_score"] = df["label_compliance_risk"]
    return df


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("escalation-classifier")

    df = build_frame()
    feats = list(FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feats], df[TARGET], test_size=0.2, random_state=42,
        stratify=df[TARGET])

    params = dict(n_estimators=250, max_depth=3, learning_rate=0.08,
                  subsample=0.9, scale_pos_weight=float(
                      (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
                  random_state=42)

    with mlflow.start_run(run_name="xgb-escalation"):
        model = xgb.XGBClassifier(**params, eval_metric="logloss")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "auc": roc_auc_score(y_test, proba),
            "recall": recall_score(y_test, pred),
            "precision": precision_score(y_test, pred),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            prov = Path(tmp) / "feature_provenance.json"
            prov.write_text(json.dumps(FEATURES, indent=2))
            mlflow.log_artifact(str(prov))

        mlflow.xgboost.log_model(model, name="model")
        MODELS_DIR.mkdir(exist_ok=True)
        model.save_model(MODELS_DIR / "escalation_classifier.json")
        print(" ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
