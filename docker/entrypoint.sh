@"
#!/bin/sh
set -e

: "${HISTORY_DIR:=/data/history}"
: "${DATA_CACHE_DIR:=/data/data_cache}"
: "${MODEL_ARTIFACTS_DIR:=/data/models}"
: "${MLFLOW_ARTIFACT_ROOT:=/mlflow/artifacts}"

mkdir -p "$HISTORY_DIR" "$DATA_CACHE_DIR" "$MODEL_ARTIFACTS_DIR" "$MLFLOW_ARTIFACT_ROOT"

exec "$@"
"@ | Out-File -FilePath .\entrypoint.sh -Encoding utf8NoBOM
