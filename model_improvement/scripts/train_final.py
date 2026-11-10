import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "model_improvement"
MODELS = OUTPUT / "models"
REPORTS = OUTPUT / "reports" / "final"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

baseline = joblib.load(ROOT / "baseline/models/model.joblib")
features = list(baseline.feature_names_in_)

train = pd.read_parquet(ROOT / "baseline/data/train.parquet")
test = pd.read_parquet(ROOT / "baseline/data/test.parquet")

X_train = train[features]
y_train = train["price"]
X_test = test[features]
y_test = test["price"]

candidate = clone(baseline).set_params(l2_regularization=100.0)

print("Training selected", flush=True)
candidate.fit(X_train, y_train)


def evaluate(model):
    predictions = model.predict(X_test)
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


baseline_metrics = evaluate(baseline)
candidate_metrics = evaluate(candidate)

improvement = {
    metric: (
        (baseline_metrics[metric] - candidate_metrics[metric])
        / baseline_metrics[metric]
        * 100
    )
    for metric in ("mae", "rmse")
}

rmse_improved = candidate_metrics["rmse"] < baseline_metrics["rmse"]

result = {
    "selected_variant": "stronger_l2",
    "selection_metric": "validation_rmse",
    "parameter_changes": {"l2_regularization": 100.0},
    "train_rows": len(train),
    "test_rows": len(test),
    "features": features,
    "sklearn_version": sklearn.__version__,
    "baseline_test": baseline_metrics,
    "candidate_test": candidate_metrics,
    "error_reduction_percent": improvement,
    "test_rmse_improved": rmse_improved,
}

model_path = MODELS / "candidate_l2_100.joblib"
joblib.dump(candidate, model_path)

with (REPORTS / "test_comparison.json").open(
    "w", encoding="utf-8"
) as stream:
    json.dump(result, stream, indent=2, ensure_ascii=False)

pd.DataFrame(
    [baseline_metrics, candidate_metrics],
    index=["baseline", "candidate_l2_100"],
).to_csv(REPORTS / "test_comparison.csv", index_label="model")

print("\nTEST COMPARISON:")
print(
    pd.DataFrame(
        [baseline_metrics, candidate_metrics],
        index=["baseline", "candidate_l2_100"],
    ).to_string()
)
print("\nERROR REDUCTION (%):")
print(json.dumps(improvement, indent=2))
print("\nTEST_RMSE_IMPROVED:", rmse_improved)
print("Saved.")