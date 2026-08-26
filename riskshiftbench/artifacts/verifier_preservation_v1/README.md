# Prospective verifier-preservation study

This directory is reserved for the main-track extension. The design compares
three task-IUT certification targets across 36 fixed agent updates:

- `absolute`: utility, downside, and absolute high-confidence failure;
- `operational`: absolute plus fixed-threshold operational drift;
- `preserve`: operational plus development-matched selective drift and a
  retained-coverage guard.

The update set crosses two model families, three agent domains, and six update
mechanisms. Coding uses the locked ten-step environment. Natural and
verifier-stress cohorts each contain 18 updates. The target budget is fixed at
2,000 paired episodes per update.

Evidence opens in this order:

```powershell
python -m riskshiftbench.experiments.verifier_preservation_study prepare-development
python -m riskshiftbench.experiments.verifier_preservation_study run-development
python -m riskshiftbench.experiments.verifier_preservation_study analyze-development
python -m riskshiftbench.experiments.verifier_preservation_study lock-target
python -m riskshiftbench.experiments.verifier_preservation_study run-target
python -m riskshiftbench.experiments.verifier_preservation_study analyze-target
```

Development inference fixes the confidence calibration and policy-specific
coverage thresholds. The target manifest then freezes those quantities and
fresh pilot, reference, and final tasks before target inference. The scripts
reject source, configuration, manifest, or raw-trajectory hash mismatches.

Do not replace an update, prompt, task, checkpoint, coverage level, boundary,
or analysis after any corresponding outcomes have been opened. Amendments must
be new versioned protocols and must retain this record.
