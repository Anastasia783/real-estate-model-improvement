import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "model_improvement" / "reports"
OUTPUT.mkdir(parents=True, exist_ok=True)

reference = joblib.load(ROOT / "baseline/models/model.joblib")
data = pd.read_parquet(ROOT / "baseline/data/train.parquet")

features = list(reference.feature_names_in_)
assert "price" not in features
assert "flat_id" not in features
assert "building_id" not in features

X = data[features]
y = data["price"]

X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.2,random_state=42,
)

variants = {
    "baseline_retrained": {},
    "fewer_iterations": {
        "max_iter": 100,
    },
    "smaller_trees": {
        "max_leaf_nodes": 15,
    },
    "larger_leaves": {
        "min_samples_leaf": 100,
    },
    "stronger_l2": {
        "l2_regularization": 100.0,
    },

}

results = []

for name, changes in variants.items():
    print(f"\nTRAINING: {name}", flush=True)
    model = clone(reference).set_params(**changes)

    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    predictions = model.predict(X_valid)

    result = {
        "variant": name,
        "mae": float(mean_absolute_error(y_valid, predictions)),
        "rmse": float(root_mean_squared_error(y_valid, predictions)),
        "r2": float(r2_score(y_valid, predictions)),
        "fit_seconds": fit_seconds,
    }
    results.append(result)

    pd.DataFrame(results).to_csv(
        OUTPUT / "validation_results.csv",
        index=False,
    )
    print(json.dumps(result, indent=2), flush=True)

table = pd.DataFrame(results).sort_values("rmse")
winner = str(table.iloc[0]["variant"])

summary = {
    "selection_metric": "rmse",
    "validation_fraction": 0.2,
    "random_state": 42,
    "train_rows": len(X_train),
    "validation_rows": len(X_valid),
    "features": features,
    "variants": variants,
    "best_variant": winner,
    "best_parameter_changes": variants[winner],
    "results": results,
}

with (OUTPUT / "experiment_summary.json").open(
    "w", encoding="utf-8"
) as stream:
    json.dump(summary, stream, indent=2, ensure_ascii=False)

print("VALIDATION RESULTS:")
print(table.to_string(index=False))
print("BEST:", winner)
print("not used")