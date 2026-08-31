#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null || {
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 2
}
command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is required on the A100 host" >&2
  exit 2
}

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "== Python environment =="
uv sync --frozen --group dev --group inference
uv pip install \
  --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu124 \
  'torch==2.6.0'

echo "== Locked model revisions =="
export HF_HUB_ENABLE_HF_TRANSFER=1
.venv/bin/hf download \
  Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
.venv/bin/hf download \
  HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --revision 31b70e2e869a7173562077fd711b654946d38674 \
  --local-dir .model-cache/SmolLM2-1.7B-Instruct-31b70e2

expected_smol="f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8"
actual_smol="$(sha256sum .model-cache/SmolLM2-1.7B-Instruct-31b70e2/model.safetensors | cut -d' ' -f1)"
if [[ "$actual_smol" != "$expected_smol" ]]; then
  echo "SmolLM2 weight digest mismatch: $actual_smol" >&2
  exit 3
fi

echo "== CUDA and lock validation =="
target_manifest="riskshiftbench/artifacts/verifier_preservation_v1/target_manifest.json"
if [[ ! -f "$target_manifest" ]]; then
  gzip -dc "${target_manifest}.gz" > "${target_manifest}.tmp"
  mv "${target_manifest}.tmp" "$target_manifest"
fi

.venv/bin/python - <<'PY'
import json

import torch

from riskshiftbench.experiments import verifier_preservation_study as study

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
if (major, minor) != (8, 0):
    raise SystemExit(f"expected A100 compute capability 8.0, got {major}.{minor} ({name})")

config = json.loads(study.CONFIG.read_text(encoding="utf-8"))
development_manifest = study._load_manifest(
    study.DEVELOPMENT_MANIFEST, config, "development"
)
development_raw = study._load_raw(study.DEVELOPMENT_RAW, development_manifest)
development = study._load_development(
    config, development_manifest, development_raw
)
target_manifest = study._load_manifest(study.TARGET_MANIFEST, config, "target")
if target_manifest["development_result_sha256"] != development["result_sha256"]:
    raise SystemExit("target lock uses another development result")

print(f"GPU: {name}")
print(f"source_sha256: {study.source_sha256()}")
print(f"target_manifest_sha256: {target_manifest['manifest_sha256']}")
print("A100 environment and prospective locks validated")
PY
