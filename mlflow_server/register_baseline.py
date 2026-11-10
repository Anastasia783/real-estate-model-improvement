"""Берем модель прошлого проекта, как baseline"""

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline"


load_dotenv(
    dotenv_path=ROOT / "mlflow_server" / ".env",
    override=False,
)

required_variables = [
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
]

missing = [
    name for name in required_variables
    if not os.environ.get(name)
]

if missing:
    raise RuntimeError(
        "Не заданы переменные: " + ", ".join(missing)
    )

EXPERIMENT = "real_estate_model_improvement"
REGISTERED_MODEL = "real_estate_price"

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise RuntimeError()

model_path = BASELINE / "models" / "model.joblib"
model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

expected_hash = (
    "29E218FFCD0C3812060D37ED63176ACDE"
    "4496C2B70703435CF67E2062664D17F"
).lower()

if model_hash != expected_hash:
    raise RuntimeError("Ошибка")

model = joblib.load(model_path)
test = pd.read_parquet(BASELINE / "data" / "test.parquet")
features = list(model.feature_names_in_)

X_test = test.loc[:, features]
y_test = test["price"]
predictions = model.predict(X_test)

metrics = {
    "mae": float(mean_absolute_error(y_test, predictions)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    "r2": float(r2_score(y_test, predictions)),
}

saved_report = json.loads(
    (BASELINE / "metrics" / "metrics.json").read_text(encoding="utf-8-sig")
)

if len(test) != saved_report["test_rows"]:
    raise RuntimeError("Есть отличие, вероятна ошибка")

for name, value in metrics.items():
    if not np.isclose(
        value, saved_report["model"][name], rtol=1e-8, atol=1e-8
    ):
        raise RuntimeError(
            f"Метрика {name} не совпала: "
            f"расчёт={value}, исходная={saved_report['model'][name]}"
        )

signature = infer_signature(X_test, predictions)
input_example = X_test.head(5)

pip_requirements = [
    f"{package}=={version(package)}"
    for package in [
        "mlflow",
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "joblib",
        "cloudpickle",
    ]
]

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="baseline") as run:
    mlflow.set_tags({
        "stage": "baseline",
        "source": "previous_project",
        "retrained": "false",
        "model_sha256": model_hash,
        "python_version": platform.python_version(),
    })

    mlflow.log_params(model.get_params())
    mlflow.log_params({
        "feature_count": len(features),
        "test_rows": len(test),
        "target": "price",
    })

    mlflow.log_metrics({
        f"test_{name}": value for name, value in metrics.items()
    })

    mlflow.log_dict({"features": features}, "metadata/features.json")
    mlflow.log_dict(metrics, "evaluation/recomputed_test_metrics.json")
    mlflow.log_artifacts(
        str(BASELINE / "metadata"), artifact_path="source_metadata"
    )
    mlflow.log_artifact(
        str(BASELINE / "metrics" / "metrics.json"),
        artifact_path="evaluation",
    )
    mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="code")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        registered_model_name=REGISTERED_MODEL,
    )

    print("EXPERIMENT:", EXPERIMENT)
    print("EXPERIMENT_ID:", run.info.experiment_id)
    print("RUN_ID:", run.info.run_id)
    print("REGISTERED_MODEL:", REGISTERED_MODEL)
    print("MODEL_URI:", model_info.model_uri)
    print("ARTIFACT_URI:", run.info.artifact_uri)
    print("TEST_METRICS:", metrics)