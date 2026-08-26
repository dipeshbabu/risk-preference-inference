# External confirmation v1.1 amendment

Status: **registration-ready; gate decisions and final evaluation blocked**.

This amendment is limited to a deterministic seed-metadata validation error in
the registered external confirmation evaluator. The PointGoal adapter records
the actual reset seed, which includes the task's already frozen
`layout_seed_base`. The v1 gate validator omitted that offset when reconstructing
the expected recorded seed. Candidate/fallback pairing is intact for every
pilot episode pair. No score, batch advantage, gate decision, or final outcome
was opened before this amendment was frozen.

## Files to register

Upload these exact files to a new public OSF registration linked to the base
registration at <https://doi.org/10.17605/OSF.IO/C576U>:

1. `configs/external_confirmation_seed_validation_amendment_v1_1.json`
2. `experiments/external_confirmation_evaluation_v1_1.py`

The amendment JSON SHA-256 is
`f2a9a96acd74966e2e7390be69956970a726cb6c84b5b6ba166f9a69217ff995`.
Its canonical JSON SHA-256 is
`45d01ecf5a4093bc63b50a06e1be4582561643324545f2c34483822bccba1cbd`.
The amended evaluator SHA-256 is
`a90a29764b10601766ed6e6ae7e605f0be4d9d2636ac0d9b3880ad91bb47a9ff`.

Suggested title:

> RiskShiftBench External Confirmation v1.1: Outcome-Blind Seed-Validation Amendment

Suggested description:

> This amendment corrects one deterministic metadata check in the publicly
> registered RiskShiftBench external confirmation protocol
> (doi:10.17605/OSF.IO/C576U). After both prespecified pilot allocations were
> completed, the first gate-locking command stopped before writing gate
> decisions because the PointGoal adapter records the actual environment reset
> seed, including the task's frozen layout seed, while the validator omitted
> that same frozen offset when reconstructing the expected seed. All 8,280
> candidate/fallback episode pairs are intact; all 2,520 affected PointGoal
> pair mismatches equal the prespecified layout offset exactly. This amendment
> changes only the expected seed-metadata formula. It does not change or rerun
> pilot outcomes, scoring, policies, tasks, allocations, familywise gate rules,
> routes, final seeds, estimands, or analyses. No utility, score, batch
> advantage, gate acceptance, or final-evaluation result was inspected before
> the amendment payload and the 54-file pilot artifact-set hash were frozen.

For foreknowledge, select **Authors' limited observation of the data could not
influence their analysis decisions**. Explain that only file paths, row counts,
hashes, task parameters, policy roles, recorded seed metadata, and process
health were inspected. Pilot outcome fields and gate results were not inspected.

## Finalize after public registration

After the new registration is public, create the registered wrapper with its
public URL and exact timestamp:

```powershell
uv run python -m experiments.external_confirmation_evaluation_v1_1 `
  --amendment configs/external_confirmation_seed_validation_amendment_v1_1.registration-draft.json `
  finalize-registration `
  --url "<public amendment URL>" `
  --registered-at "<registration timestamp>" `
  --output configs/external_confirmation_seed_validation_amendment_v1_1.registered.json
```

Then run `dry-run` with the registered wrapper. It must report valid base and
amended hashes and `confirmation_execution_allowed=true` before `lock-gates`,
`final`, or `combine` is invoked.
