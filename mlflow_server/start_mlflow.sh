#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

docker compose --env-file .env up --build --wait --wait-timeout 180
docker compose ps

echo "MLflow: http://localhost:5000"