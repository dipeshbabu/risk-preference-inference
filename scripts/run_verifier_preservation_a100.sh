#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Run scripts/bootstrap_verifier_preservation_a100.sh first" >&2
  exit 2
fi

artifact_dir="riskshiftbench/artifacts/verifier_preservation_v1"
log_file="$artifact_dir/a100-target.log"
mkdir -p "$artifact_dir"

max_restarts="${MAX_RESTARTS:-3}"
attempt=0
while (( attempt <= max_restarts )); do
  attempt=$((attempt + 1))
  echo "[$(date -u +%FT%TZ)] target attempt $attempt/$((max_restarts + 1))" | tee -a "$log_file"

  set +e
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -u -m riskshiftbench.experiments.verifier_preservation_study run-target \
    2>&1 | tee -a "$log_file"
  status="${PIPESTATUS[0]}"
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "[$(date -u +%FT%TZ)] target inference complete" | tee -a "$log_file"
    exit 0
  fi
  if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
    echo "[$(date -u +%FT%TZ)] target inference interrupted by user" | tee -a "$log_file"
    exit "$status"
  fi
  if (( attempt > max_restarts )); then
    echo "target inference failed after $attempt attempts" >&2
    exit "$status"
  fi

  echo "[$(date -u +%FT%TZ)] process failed with $status; checkpoint retained; retrying in 30s" \
    | tee -a "$log_file"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
    | tee -a "$log_file" || true
  sleep 30
done
