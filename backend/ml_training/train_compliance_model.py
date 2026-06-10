"""Train the compliance risk scorer (Option 2).

XGBoost regressor on rule-derived continuous risk labels. Logs params,
metrics, SHAP summary, and a feature-provenance manifest to MLflow.
The same model serves two gates at runtime: the triage cutoff (0.40) and
the senior band (0.70) — one model, two thresholds.
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
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data" / "loans" / "records.jsonl"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

LOAN_TYPE_MAP = {"equipment": 0, "working_capital": 1, "cre": 2}
EMP_MAP = {"established_business": 0, "newer_business": 1, "seasonal": 2}

# Feature provenance: doc = extracted from rendered documents (round-trip
# tested); system = portfolio history tables, never in documents.
FEATURES: dict[str, str] = {
    "ltv_ratio": "doc:application+appraisal",
    "dti_ratio": "doc:financials",
    "fico_score": "doc:credit_pull",
    "loan_amount": "doc:application",
    "loan_type_enc": "doc:application",
    "employment_type_enc": "doc:application",
    "doc_completeness": "doc:packet_inventory",
    "income_discrepancy_pct": "doc:application_vs_tax_returns",
    "state_risk_level": "system:state_reference",
    "prior_exceptions_count": "system:exception_history",
    "is_jumbo": "doc:application",
}
TARGET = "label_compliance_risk"


def load_frame() -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in open(DATA)])
    df["loan_type_enc"] = df["loan_type"].map(LOAN_TYPE_MAP)
    df["employment_type_enc"] = df["employment_type"].map(EMP_MAP)
    df["is_jumbo"] = df["is_jumbo"].astype(int)
    return df


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("compliance-risk-scorer")

    df = load_frame()
    feats = list(FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feats], df[TARGET], test_size=0.2, random_state=42)

    params = dict(n_estimators=300, max_depth=4, learning_rate=0.06,
                  subsample=0.9, colsample_bytree=0.9, random_state=42)

    with mlflow.start_run(run_name="xgb-compliance"):
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        pred = np.clip(model.predict(X_test), 0, 1)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        # Gate-level agreement: does the model put loans in the same routing
        # band as the rule-derived label? This is the metric that matters.
        bands = lambda v: np.digitize(v, [0.40, 0.70])  # noqa: E731
        band_agree = float(np.mean(bands(pred) == bands(y_test.values)))
        # Fast-track safety: of loans the model would fast-track, how many
        # actually carried a >=medium label band? Must be ~0.
        ft = pred < 0.40
        ft_miss = float(np.mean(y_test.values[ft] >= 0.40)) if ft.any() else 0.0

        mlflow.log_params(params)
        mlflow.log_metrics({"mae": mae, "r2": r2,
                            "band_agreement": band_agree,
                            "fast_track_miss_rate": ft_miss})

        with tempfile.TemporaryDirectory() as tmp:
            prov = Path(tmp) / "feature_provenance.json"
            prov.write_text(json.dumps(FEATURES, indent=2))
            mlflow.log_artifact(str(prov))
            try:
                import shap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_test)
                shap.summary_plot(sv, X_test, show=False)
                fig_path = Path(tmp) / "shap_summary.png"
                plt.savefig(fig_path, bbox_inches="tight", dpi=120)
                plt.close()
                mlflow.log_artifact(str(fig_path))
            except Exception as exc:  # pragma: no cover
                print(f"shap artifact skipped: {exc}")

        mlflow.xgboost.log_model(model, name="model")
        MODELS_DIR.mkdir(exist_ok=True)
        model.save_model(MODELS_DIR / "compliance_risk.json")

        print(f"mae={mae:.4f} r2={r2:.3f} band_agreement={band_agree:.3f} "
              f"fast_track_miss_rate={ft_miss:.4f}")


if __name__ == "__main__":
    main()
