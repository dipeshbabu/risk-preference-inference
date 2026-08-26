"""Capture the software and hardware record for the natural-agent study."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import accelerate
import torch
import transformers


OUTPUT = Path("riskshiftbench/artifacts/natural_agent_updates_v1/environment.json")


def main() -> None:
    gpu_name = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .splitlines()[0]
    )
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpu": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "runtime_notice": (
            "Qwen emitted a Transformers SDPA sliding-window notice; all prompts "
            "were capped at 640 tokens, below the model context window."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
