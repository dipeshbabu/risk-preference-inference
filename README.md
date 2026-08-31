# Operational verifier drift in agent updates

This repository contains the RiskShiftBench implementation for operational
verifier drift and prospective verifier-preservation studies.

## Active layout

```text
riskshiftbench/    VDC implementation, current experiments, tests, lock records
schemas/           Submission and manifest schemas
```

## Checks

Run the current implementation tests:

```powershell
pytest -q riskshiftbench/tests
```

Validate the completed Full-versus-Absolute experiment:

```powershell
python -m riskshiftbench.experiments.complete_full_vs_absolute_target validate
```

Prepare or resume the locked verifier-preservation development study:

```powershell
python -m riskshiftbench.experiments.verifier_preservation_study prepare-development
python -m riskshiftbench.experiments.verifier_preservation_study run-development
```

Generated trajectories and local model weights are ignored by default. Compact
configuration, feasibility, and lock records remain versioned.

## A100 target execution

The verifier-preservation target is locked to the versioned source, development
evidence, and target manifest. On a Linux A100 host, bootstrap the pinned CUDA
environment and model revisions with:

```bash
bash scripts/bootstrap_verifier_preservation_a100.sh
```

Run the target inside `tmux` or another persistent terminal:

```bash
bash scripts/run_verifier_preservation_a100.sh
```

The launcher preserves the declared batch sizes and resumes only completed
physical runs. It retries a failed process from the latest run-level checkpoint;
it never deletes or rewrites a valid checkpoint. A fresh clone starts the target
phase from zero on one hardware class, while the development evidence and target
lock are already available for validation.
