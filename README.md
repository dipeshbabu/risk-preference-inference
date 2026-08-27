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
