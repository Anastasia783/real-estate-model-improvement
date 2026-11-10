import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    ParameterSampler,
)
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "model_improvement/reports/search_cv"
MODELS = ROOT / "model_improvement/models"

REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_iter": [100, 200, 400],
    "max_leaf_nodes": [7, 15, 31],
    "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [10.0, 30.0, 100.0, 300.0, 1000.0],
}


def main():
    reference = joblib.load(ROOT / "baseline/models/model.joblib")
    train = pd.read_parquet(ROOT / "baseline/data/train.parquet")

    features = list(reference.feature_names_in_)
    if any(name in features for name in ("price", "flat_id", "building_id")):
        raise ValueError("Unexpected target or ID in baseline features.")

    X = train[features]
    y = train["price"]

    baseline_params = {
        name: reference.get_params()[name]
        for name in SEARCH_SPACE
    }
    previous_candidate = {
        **baseline_params,
        "l2_regularization": 100.0,
    }

    sampled = list(
        ParameterSampler(
            SEARCH_SPACE,
            n_iter=24,
            random_state=42,
        )
    )

    candidates = []
    seen = set()

    for params in [baseline_params, previous_candidate, *sampled]:
        identity = tuple(sorted(params.items()))
        if identity not in seen:
            seen.add(identity)
            candidates.append(params)

    parameter_grid = [
        {name: [value] for name, value in params.items()}
        for params in candidates
    ]

    cv = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=clone(reference),
        param_grid=parameter_grid,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        refit="rmse",
        cv=cv,
        n_jobs=1,
        verbose=2,
        error_score="raise",
        return_train_score=False,
    )

    plan = {
        "selection_metric": "mean_cv_rmse",
        "cv": {
            "type": "KFold",
            "n_splits": 3,
            "shuffle": True,
            "random_state": 42,
        },
        "sampler_random_state": 42,
        "search_space": SEARCH_SPACE,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "train_rows": len(train),
        "features": features,
        "sklearn_version": sklearn.__version__,
        "test_used": False,
    }

    with (REPORTS / "search_plan.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False)

    print("CANDIDATES:", len(candidates), flush=True)
    print("CV FITS:", len(candidates) * 3, flush=True)
    print("Test data will not be read.", flush=True)

    with threadpool_limits(limits=2):
        search.fit(X, y)

    raw = pd.DataFrame(search.cv_results_)
    raw.to_csv(REPORTS / "cv_results_full.csv", index=False)

    table = pd.DataFrame({
        "rank": raw["rank_test_rmse"],
        "cv_rmse": -raw["mean_test_rmse"],
        "cv_rmse_std": raw["std_test_rmse"],
        "cv_mae": -raw["mean_test_mae"],
        "cv_r2": raw["mean_test_r2"],
        "fit_seconds": raw["mean_fit_time"],
    })

    for name in SEARCH_SPACE:
        table[name] = raw["param_" + name]

    table["variant"] = [
        (
            "baseline"
            if params == baseline_params
            else "previous_l2_100"
            if params == previous_candidate
            else "sampled"
        )
        for params in raw["params"]
    ]

    table = table.sort_values(["rank", "cv_mae"])
    table.to_csv(REPORTS / "cv_results.csv", index=False)

    best_index = search.best_index_
    summary = {
        **plan,
        "best_params": search.best_params_,
        "best_cv_rmse": float(-search.best_score_),
        "best_cv_rmse_std": float(
            raw.iloc[best_index]["std_test_rmse"]
        ),
        "best_cv_mae": float(
            -raw.iloc[best_index]["mean_test_mae"]
        ),
        "best_cv_r2": float(
            raw.iloc[best_index]["mean_test_r2"]
        ),
        "model_refit_on_full_train": True,
    }

    with (REPORTS / "best_summary.json").open("w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    joblib.dump(
        search.best_estimator_,
        MODELS / "candidate_cv_best.joblib",
    )

    print("\nTOP 10:")
    print(table.head(10).to_string(index=False))

    print("\nREFERENCE MODELS:")
    print(
        table[table["variant"] != "sampled"]
        .to_string(index=False)
    )

    print("\nBEST_PARAMS:")
    print(json.dumps(search.best_params_, indent=2))
    print("\nBEST_CV_RMSE:", -search.best_score_)
    print("Saved candidate_cv_best.joblib.")
    print("Previous models were not modified. Test data was not used.")


if __name__ == "__main__":
    main()