# Operational verifier drift in agent updates

This repository contains the RiskShiftBench implementation for operational
verifier drift and prospective verifier-preservation studies. Manuscript
sources and rendered paper files are intentionally excluded from this code PR.

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

Large raw trajectories, local model weights, and manuscript contents are kept
outside Git. Compact configuration, feasibility, and lock records remain in
the repository.
