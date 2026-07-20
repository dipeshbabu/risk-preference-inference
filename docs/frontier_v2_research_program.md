# RiskShiftBench v2 research program

Working title: **Anytime-Valid Familywise Policy Routing Under Distribution
Shift**

Status: development-only design. This document does not amend, reinterpret, or
extend the completed v1 registration. No v2 confirmation task may be reset
until a separate machine-readable design has been publicly registered.

## Why v2 needs a new method

The v1 external study made the deployment rule prospective and familywise, but
its mathematical claim is about nine independent batch signs. It does not
directly test the mean score effect, cannot stop early, spends the same pilot
budget on easy and hard proposals, and does not bound the expected improvement
of the deployed router. The external evidence also spans only three domains.

Adding tasks without changing the method would improve breadth but would not
make the contribution substantially deeper. V2 therefore has two independent
requirements:

1. a stronger deployment theorem; and
2. a broader, budget-matched external evaluation.

## Proposed statistical object

For proposal \(i\), let \(\theta_i\) be the candidate-minus-fallback mean
score difference under the registered confirmation-task distribution, and let
\(X_{i,t}\in[a_i,b_i]\) be the paired difference from the next pilot unit.
Scores and bounds are fixed before confirmation. The deployment margin is
\(\delta_i\geq 0\), and the task-level deployment null is

\[
H_i:\quad \theta_i\leq\delta_i.
\]

The registered sampling link is that, whenever \(H_i\) is true, every next
pilot unit obeys
\(\mathbb{E}[X_{i,t}\mid\mathcal{F}_{t-1}]\leq\delta_i\). This holds under
the intended fixed task distribution with fresh, disjoint episode seeds; more
generally it is the substantive bridge from sequential pilot evidence to the
deployment target. Stating the link separately avoids treating the complement
of a time-varying conditional-mean statement as a deployment parameter.

For a fixed betting fraction \(\lambda\geq0\), Hoeffding's lemma gives the
test supermartingale

\[
E_{i,n}(\lambda)=
\exp\left\{
\lambda\sum_{t=1}^{n}(X_{i,t}-\delta_i)
-\frac{n\lambda^2(b_i-a_i)^2}{8}
\right\}.
\]

V2 mixes a prespecified finite grid of betting fractions. A convex mixture of
nonnegative test supermartingales remains a test supermartingale. Proposal
\(i\) is deployed when its mixture e-process reaches \(1/\alpha_i\), where the
task weights are frozen from development evidence and satisfy
\(\sum_i\alpha_i\leq\alpha\).

## Primary theorem target

Assume that every next paired observation remains within its registered bounds
and satisfies the task-level conditional-mean null when that null is true.
Tasks may be interleaved using any predictable, outcome-adaptive allocation
rule, and sampling may stop at arbitrary data-dependent times. Then

\[
\Pr\{\text{at least one proposal with }\theta_i\leq\delta_i
\text{ is deployed}\}
\leq\sum_i\alpha_i\leq\alpha.
\]

Proof sketch: each task mixture is an e-process under its null. Optional
skipping preserves the supermartingale property under predictable adaptive
allocation. Ville's inequality bounds the probability that task \(i\) ever
crosses \(1/\alpha_i\) by \(\alpha_i\). A union bound completes the result. No
independence assumption across tasks is needed; the conditional validity of
the next within-task observation is the substantive assumption.

On the same event, if fallback deployment is defined to have difference zero,
the expected score improvement of the deployed router on the fixed task family
obeys

\[
\sum_i w_i R_i\theta_i
>\sum_i w_i R_i\delta_i\geq0,
\]

where \(R_i\) records candidate deployment. This is a familywise lower bound
on expected improvement over fallback, not an oracle-regret guarantee and not
a guarantee about every finite final sample.

The reference implementation is
`experiments/anytime_familywise_router.py`. It contains two e-processes. A
finite Hoeffding mixture is the conservative validity baseline. The primary
development method is a mixture of bounded betting processes. After mapping
the observation to \(Y_{i,t}\in[0,1]\) with null mean \(q_i\), each component
updates as

\[
B_{i,n}(\lambda)=\prod_{t=1}^{n}
\{1+\lambda(Y_{i,t}-q_i)\},
\qquad 0\leq\lambda<1/q_i.
\]

The betting fractions are fixed before sampling, every factor is nonnegative,
and the conditional expected factor is at most one under the null. Mixing the
components therefore preserves anytime validity while adapting evidence growth
to effect size.

The implemented predictable plug-in comparison uses a frozen mixture of prior
strengths. Before observation (t), each component forms a null-centered
smoothed estimate from observations (1{:}t-1), uses the corresponding
one-sided Kelly fraction, and clips it below one. The resulting stake is
predictable and every multiplicative factor remains positive, so the same
conditional supermartingale argument applies. It is compared under both
uniform and adaptive resolution allocation.

The comparison suite now also includes an IID-only stitched Bentkus racing
rule. It computes the Bentkus--Pinelis binomial `P2` bound by enumerating the
analytic minimizer on every support interval, uses maximal finite-horizon
boundaries, and stitches horizons with geometric spacing 2 and weights
`h(k)=(k+1)(k+2)`, whose reciprocal sum is one. Exact dynamic programming
checks the null crossing probability through horizon 128. This is deliberately
labeled as an independent-task-stream comparator: the cited construction
assumes independence and therefore does not support the primary router's
stronger conditional-mean claim under arbitrary predictable interleaving.

- [Near-Optimal Confidence Sequences for Bounded Random Variables](https://proceedings.mlr.press/v139/kuchibhotla21a.html)
- [Estimating Means of Bounded Random Variables by Betting](https://arxiv.org/abs/2010.09686)
- [Familywise Error Rate Control by Interactive Unmasking](https://proceedings.mlr.press/v119/duan20d.html)

### Closest-method boundary

Adaptive Learn-then-Test (aLTT) already established e-process-based adaptive
acquisition, anytime stopping, and FWER-controlled online policy selection.
Those ingredients are therefore prior art, not v2 novelty. V2's distinct
statistical object is a *paired candidate-minus-fallback task effect*, followed
by deployment of a task-indexed router rather than certification of an
absolute-risk hyperparameter set. Its additional claims are the
fallback-relative deployed-router improvement bound and a registered
gap-dependent resolution-cost certificate. Empirically, v2 asks whether those
additions help across a heterogeneous, independently maintained nine-domain
suite under a prospective confirmation split; aLTT used 20 TD3+BC policies in
one HalfCheetah environment as its policy-selection example.

Two policy-improvement lines further narrow the claim. CSPI-MT already tests
multiple candidate policies against a baseline and targets expected realized
improvement, but it is an asymptotic, sample-split procedure for threshold
policies. N-SCORE already supplies finite-sample anytime-valid paired policy
comparison for bounded nonbinary metrics and uses Bonferroni in its multi-policy
experiments. Sequential paired policy comparison is therefore also prior art.
RiskShiftBench v2 instead targets simultaneous task-indexed deployment under
outcome-adaptive allocation, a conditional-mean validity model, and a frozen
family-level resolution-cost certificate.

To make that boundary testable,
`experiments/familywise_close_comparators.py` implements an aLTT reproduction
from the published specification, with aGRAPA betting, running-maximum anytime p-values, Bonferroni
FWER control, and the paper's epsilon-greedy acquisition at epsilon 0.25. The
bet truncation is frozen at 0.95 and the regularized predictable mean and
variance estimators are stated in code. It also implements the 2026 e-Holm
closed-test rule on the same current task e-values. E-Holm has always-valid
strong FWER control and can strictly dominate inverse-e Bonferroni/Holm
aggregation; it is a close comparator, not a v2 contribution. The primary v2
router retains frozen task weights and its deployment-cost certificate because
the current unweighted e-Holm result does not supply that certificate.
The same module implements an 11-bin N-SCORE reproduction. It uses the
paper's outer-marginal histogram and maximizes the resulting empirical expected
log-growth before each observation. The exact paired difference, rather than
the histogram approximation, updates the e-process. An anytime-Bonferroni,
epsilon-greedy familywise lift makes N-SCORE budget-comparable in the synthetic
task family; this lift is a comparator, not a claim made by the N-SCORE paper.

- [Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection](https://proceedings.mlr.press/v267/zecchin25a.html)
- [Family-wise Error Rate Control with E-values](https://arxiv.org/abs/2501.09015)
- [CSPI-MT: Calibrated Safe Policy Improvement with Multiple Testing for Threshold Policies](https://arxiv.org/abs/2408.12004)
- [Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison](https://arxiv.org/abs/2603.13616)

Every synthetic calibration and paired method-comparison artifact is now bound
to the newline-canonicalized statistical implementation digest
`d7d078b95fb27d268719c3fccc2e8c314403d91ff25e448c378556366881fa27`.
The digest covers the router, calibration generator, valid comparison methods,
paired comparison runner, and the hash definition itself. The readiness audit
requires current-digest artifacts for at least 10,000 primary global-null
families, 10,000 predictable-comparator global-null families under both
allocations, and a 300-trial comparison covering every declared method. Null
calibration additionally requires 10,000 matched global-null families for the
aLTT, e-Holm, and N-SCORE close comparators. Null calibration is considered adequate only
when the Wilson 95% upper bound remains
at or below 0.05. No observed power or utility threshold is used to select a
winner.

## Adaptive pilot allocation

Validity must not depend on the allocation heuristic. The first implementation
therefore separates the e-process from sampling. It includes:

- uniform round-robin allocation; and
- a resolution heuristic that forces initial coverage, estimates the remaining
  evidence needed for either decision, and samples the proposal predicted to
  resolve next.

The primary development scheduler now adds a certified betting allocation.
Before confirmation, each task receives a score-scale planning gap
\(\Delta_i>0\) derived only
from development evidence and a task-level resolution-failure budget
\(\beta_i\), with \(\sum_i\beta_i\leq\beta\). For a standardized paired
difference \(Z_t\in[-1,1]\), let \(d_i=\Delta_i/r_i\), where \(r_i\) is the
radius of the symmetric registered score-difference interval. For a fixed
betting fraction \(f\in(0,1)\) and a conditional standardized mean gap of at
least \(d\) in either direction, concavity gives the per-observation expected
log-growth bound

\[
g_f(d)=\frac{1}{2}\log(1-f^2)
+\frac{d}{2}\log\left(\frac{1+f}{1-f}\right).
\]

The range of the log increment is
\(h_f=\log((1+f)/(1-f))\). Conditional Hoeffding--Azuma concentration
therefore implies, with probability at least \(1-\beta_i\),

\[
\sum_{t=1}^n\log(1+fZ_t)
\geq ng_f(d_i)-h_f\sqrt{n\log(1/\beta_i)/2}.
\]

If the component has frozen mixture weight \(v_f\), its task e-process has
crossed \(1/\alpha_i\) once the right-hand side reaches
\(\log(1/(\alpha_i v_f))\). The implementation solves this quadratic in
\(\sqrt n\) for every frozen betting fraction and uses the smallest integer
bound \(n_i^*\). The scheduler forces initial coverage and then completes the
smallest remaining registered quota first. If no per-task cap truncates a
quota and the global budget is at least \(\sum_i n_i^*\), every task whose
conditional mean is separated from the margin by at least its planning gap is
resolved with probability at least \(1-\sum_i\beta_i\). No independence across
tasks is used.

Planning gaps affect only this cost guarantee. If they are optimistic, the
familywise false-deployment theorem still holds because acceptance continues
to use the original e-process and alpha thresholds; the affected task may
simply remain unresolved. `CertifiedSampleTarget` exposes both the theoretical
quota and the scheduled quota and marks every cap-truncated target. Efficiency
is evaluated at identical candidate-plus-fallback episode budgets.

The first transparent reference is now implemented in
`experiments/familywise_policy_baselines.py`. For task \(i\), it spends the
one-sided error budget over sample sizes as

\[
\gamma_{i,n}=\frac{6\alpha_i}{\pi^2n^2},\qquad
r_{i,n}=(b_i-a_i)
\sqrt{\frac{\log(1/\gamma_{i,n})}{2n}}.
\]

Hoeffding's inequality and

\[
\sum_{n=1}^{\infty}\gamma_{i,n}=\alpha_i
\]

give a time-uniform lower confidence sequence
\(\bar X_{i,n}-r_{i,n}\) with task-level error at most \(\alpha_i\).
A separate upper sequence supports safe futility decisions. The racing rule
samples an unresolved task with the widest interval and removes a task after
either boundary crosses the deployment margin. On the simultaneous coverage
event, a task with gap \(\Delta_i=|\mu_i-\delta_i|>0\) is resolved no later
than the first \(n\) satisfying \(2r_{i,n}<\Delta_i\), subject to its frozen
per-task cap. Thus, when the global cap is nonbinding, the total sample count
is at most the sum of these task-specific resolution bounds. This baseline is
deliberately conservative, but it supplies an implementation-checkable
familywise proof and a concrete gap-dependent cost statement against which
the betting allocation can be judged.

## External proposal lock

`experiments/frontier_v2_router_lock.py` deterministically converts only the
complete 20-episode development and calibration suites into the external
confirmation plan. For each domain, the candidate with the largest equal-task
paired mean development effect is selected; exact ties follow the frozen
candidate order. All four confirmation tasks in every domain remain in the
family, so weak or negative development evidence cannot silently remove a
domain from the breadth claim.

The corresponding calibration task at the same within-domain index supplies
an allocation-only planning gap. Its absolute paired mean effect is clipped to
[0.05, 0.50]. A wrong gap can make the scheduler inefficient, but it cannot
change an e-process or an acceptance threshold. The 36 proposals receive equal
weights, familywise acceptance and futility budgets of 0.05, a zero deployment
margin, two forced initial observations, a 200-observation task cap, and the
certified betting allocation.

The shared primary pilot cap is 3,960 paired observations, or 7,920
candidate-plus-fallback policy episodes. This exactly matches the v1
proposal-focused pilot episode budget while covering the larger nine-domain
family. The 14 routing and testing configurations are compared before
registration in the frozen 23-task synthetic family, where every method
receives the same 1,150-observation budget and the same task-indexed latent
streams. They are not rerun on the confirmation tasks and cannot consume or
influence confirmation outcomes. The readiness audit reconstructs the lock
from all 72 input artifacts and requires byte-equivalent content before a v2
registration can be created; it still never authorizes confirmation execution.

`experiments/frontier_v2_protocol_lock.py` then refuses to create a registration
draft until the source, statistical, rehearsal, router, 12 learned-baseline, and
six nonlearned-baseline gates all pass. The draft hashes the complete design,
the router lock, every baseline manifest, the current statistical artifacts,
and the protocol-specific source. A second wrapper can be finalized only with
an immutable HTTPS registration URL and timezone-aware registration time.

`experiments/frontier_v2_confirmation_runtime.py` remains inert without that
finalized wrapper. Its pilot log is append-only and hash-chained. On every
request, both the parent and the isolated codebase worker replay the complete
log, reconstruct the e-processes, and require the request to equal the unique
next task, policy pair, and canonical seed under the registered scheduler. A
crash can therefore repeat the same deterministic pair but cannot skip ahead
or silently change allocation. Persistent workers use the separately pinned
Python environments for Gymnasium, OR-Gym, Safety-Gymnasium, and MiniGrid.

The final stream is blocked until the pilot terminates and its decision file
exactly matches the authenticated log. It evaluates the frozen candidate and
fallback on 100 common-random-number episodes per task using seed blocks
disjoint from the pilot.

The sole confirmatory test targets the realized, pilot-frozen router on the
fixed 36-task family. Let \(D_{dij}\in[-1,1]\) be final paired episode
\(j\) for task \(i\) in domain \(d\), and let \(R_{di}\) be the frozen
candidate-route indicator. With nine domains, four tasks per domain, and 100
episodes per task, the registered estimator is

\[
\widehat\Theta=\sum_{d,i,j}c_{dij}D_{dij},\qquad
c_{dij}=\frac{R_{di}}{9\cdot4\cdot100}.
\]

Conditional on the frozen route, the final episode pairs use distinct seeds
and are assumed independent; their distributions need not be identical. If
\(S=\sum_{d,i,j}(2c_{dij})^2\), the one-sided weighted Hoeffding p-value for
\(H_0:\Theta\leq0\) is
\(\exp\{-2\max(\widehat\Theta,0)^2/S\}\), with p-value one when no
candidate is deployed. The matching 95% lower confidence bound is
\(\widehat\Theta-\sqrt{S\log(20)/2}\), truncated only by the feasible lower
bound from the accepted task weight. Confirmation requires a positive estimate
and p-value at most 0.05.

The deterministic analysis additionally reports fixed-domain task and
domain-resampling bootstraps, a symmetry-based task sign-flip sensitivity,
candidate-everywhere effect, harmful accepted routes, per-domain effects, and
leave-one-domain-out effects. These are secondary and do not replace the
bounded confirmatory test. All outcome and analysis files are write-once and
checked against frozen hashes on resume.

## Required baselines

Every comparison must receive the same pilot episode budget and the same
candidate/fallback pairs.

1. fallback only;
2. candidate everywhere;
3. fit-only routing;
4. v1 fixed unanimity with Bonferroni;
5. fixed-sample Bonferroni mean test;
6. fixed-sample Holm step-down test;
7. anytime mixture e-process with uniform allocation;
8. anytime mixture e-process with active allocation;
9. an outcome-blind random-task allocation;
10. an IID-only stitched Bentkus racing rule;
11. aLTT with aGRAPA betting, anytime Bonferroni, and epsilon-greedy
    acquisition;
12. always-valid e-Holm with the same aGRAPA evidence and acquisition; and
13. N-SCORE with 11-bin expected-log-growth betting, anytime Bonferroni,
    and epsilon-greedy acquisition; and
14. a robust test-selection baseline inspired by
   [RPOSST](https://proceedings.mlr.press/v216/morrill23a.html).

The primary comparisons are familywise-valid methods. Candidate-everywhere and
fit-only routing remain descriptive upper-recall references.

`experiments/familywise_policy_comparison.py` implements paired synthetic
comparisons for fixed-sample Hoeffding tests with Bonferroni or Holm correction,
fixed-sample sign tests with Bonferroni or Holm correction, the alpha-spending
racing rule, the Hoeffding-mixture router, and the betting-mixture router. The
sign methods test an independent sign null rather than the conditional-mean
null and are labeled separately; they are not interchangeable guarantees.
The comparison now includes 15 statistical routing/testing configurations in
total because allocation variants are counted separately. A first 100-family
development smoke comparison found 4/100 global-null families with any false
acceptance for both aLTT and e-Holm; its Wilson upper bound is too wide for the
readiness gate. In a separate 100-family mixed-effect smoke comparison, the
certified v2 router accepted 39.0% of positive tasks versus 20.4% for aLTT at
the same 1,150-observation budget, with no versus one false-accept family.
These small diagnostics justify retaining the comparison but are not method
selection evidence; the registered method cannot be chosen from them.
The N-SCORE lift separately produced two false-accept families in a 100-family
global-null smoke run and accepted 4.0% of positive tasks in a 100-family mixed
run. Its null interval is again too wide for readiness; the result is retained
only as an implementation diagnostic pending the locked 10,000-family run.

The learned-policy reference protocol is now machine-readable in
`experiments/frontier_v2_baseline_design.py`. Every learned baseline uses five
frozen training seeds, retains checkpoints every 50,000 environment steps, and
uses development tasks for training and exactly 100 episodes per calibration
task for checkpoint selection. Non-safe policies maximize equal-task normalized
score. PPO-Lagrangian and CPO first require equal-task mean episode cost at most
25, with a prespecified minimum-cost fallback if no checkpoint is feasible.
The training budgets are 500,000 steps per seed for tabular Q-learning and one
million for every deep or safe-RL reference.

External implementations are pinned to OmniSafe commit
`15603dd7a654a991d0a4648216b69d60b81a6366`, MiniGrid's recommended
`rl-starter-files` commit
`317da04a9a6fb26506bbd7f6c7c7e10fc0de86e0`, and CleanRL commit
`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`. The current competitive-baseline
design hash is
`60cde38aa5210406d74f5692bcb50cca9c25a5b45c0b2034db1f02bbd2f84d95`,
and the internal trainer implementation hash is
`0900d4fd795d3ae354bd04d7ed4a09c0798444a923c6d1e0daa6f98f510f027e`.
These remain development hashes until the complete baseline suite is locked.

`experiments/frontier_v2_baseline_source_audit.py` now verifies all three
physical checkouts, exact commits, clean working trees, declared license text,
and the algorithm entry points before a baseline can be treated as ready. The
audited implementation hashes are `74dafd200ed8494b2653c32882d659369d58414e7d4987da774716eb2d7a3670`
for OmniSafe PPO-Lagrangian,
`38e6417b1ebe3b3e87923c6a6af3b59218844035ebab895efb43159ae55a8783`
for OmniSafe CPO,
`f49a075ca2dc722f101486da3186065cea810ddf7ba04c1451edd358e06eedcd`
for the MiniGrid PPO training entry point,
`b61592b8cf909a8a88081498daedf111b2abdd33affc6c51bd71c6fe6632b26e`
for its recurrent actor-critic model, and
`a0ea7da3c80d56c0701d3d36348887c7d5c04b8a2ab799d2a421458f809f8743`
and `6f4c04e3349f5b6a1ebbd29775416da80344f8cb5b6368952017d8898bc1ebe8`
for CleanRL DQN and PPO. CleanRL's pinned file implements ordinary DQN; the
prespecified Double-DQN reference therefore requires a repository-local,
hashed target-selection adaptation and must not be described as an unchanged
upstream run.

The five-seed, 500,000-step tabular Q-learning references are complete for
CliffWalking and Taxi. `experiments/frontier_v2_baseline_audit.py` verified all
100 physical checkpoint files, their SHA-256 hashes, complete schedules, source
and design locks, and the mechanical selection rule. CliffWalking selected
calibration scores ranged from 0.3770 to 0.7876 across seeds, with selected
steps from 300,000 to 450,000. Taxi scores ranged from 0.9611 to 0.9615, with
selected steps from 200,000 to 500,000. All five seeds will be reported; the
best seed is not substituted for the prespecified replicate distribution.
Deep OR-Gym and MiniGrid references and both five-seed Safety-Gymnasium
PPO-Lagrangian/CPO references remain incomplete gates.

The six nonlearned references are now executable rather than design-table
labels. `experiments/frontier_v2_nonlearned_baselines.py`, bound to digest
`ea3d729815868eac5c431d522fcdb75cc5afff7a8b0d591c81daaa4d8d962eec`,
ran 100 canonical calibration episodes per task and repeated every trajectory
exactly. The FrozenLake, CliffWalking, and Taxi tabular oracles solve the exact
finite-horizon dynamic program for the frozen bounded score, augmenting state
by time and accumulated penalties. Their equal-task calibration means were
0.8303, 0.8677, and 0.9613. The online-knapsack reference is explicitly
nondeployable: it computes the fractional-knapsack upper bound after observing
the complete realized item sequence; its mean normalized score was 0.5650.
The inventory reference applies a lead-time Poisson newsvendor critical-ratio
base-stock rule and scored 0.4756.

The transferred v1 FrozenLake router was reconstructed from the exact archived
development and calibration aggregates and checked against the archived router
report. Each v2 task is converted to the v1 feature schema of map size,
slipperiness, and transition success rate; v2 map density is not substituted
for the old success-rate coordinate. Without refitting or reading v2 outcomes,
the router selected the nominal policy on two calibration tasks and the
hazard-averse policy on two, for an equal-task mean score of 0.8289. All six
manifests pass source-lock, task-hash, seed-block, score-bound, derived-summary,
input-hash, and disk-round-trip replay audits.

The pooled-development Double-DQN adapter is now implemented in
`experiments/frontier_v2_double_dqn.py` and bound to implementation digest
`9ed99d797067a3bdc1bd556a5167c05cb07456ed6068eaf4111a2e01f22a8816`.
It preserves the pinned CleanRL MLP/replay structure, adds explicit online
action selection with target-network evaluation, respects action masks, cycles
evenly across all four development tasks, and evaluates frozen checkpoints on
calibration tasks only. Real CUDA smoke runs crossed the 10,000-step learning
start and completed at 12,000 steps for online knapsack, DynamicObstacles, and
LavaCrossing. Their two-episode-per-task calibration scores were 0.442, 0.000,
and 0.000, respectively. These are pipeline diagnostics, not baseline results;
all five one-million-step seeds remain required.
The smoke-selected LavaCrossing checkpoint was also reconstructed in a fresh
network and reproduced its stored calibration score and cost exactly. Full
runs apply this replay audit to the selected checkpoint from every seed.

`experiments/frontier_v2_ppo.py` now supplies the remaining inventory and
MiniGrid PPO execution path, bound to digest
`c821fc5d438036f2894387ea28f550b97405acf912978550532f901cf29351e5`.
Inventory uses a frozen 39-coordinate padded state, three independent
11-category order heads mapped across each stage's physical supply capacity,
and a clipped PPO objective. MiniGrid uses a 128-state LSTM, ten parallel
development environments, per-environment GAE, and five-step truncated
backpropagation. Both use 250 transitions per update, four update epochs, and
calibration-only checkpoint selection. A 1,000-transition inventory smoke
scored 0.454 across two calibration episodes per task; recurrent
DynamicObstacles remained at zero at the same intentionally tiny budget. Both
selected checkpoints reproduced their score and cost exactly after reload.
The smoke runtimes show that the inventory reference is locally tractable but
the five-seed recurrent references should run on stronger recorded hardware;
this is a compute-planning conclusion, not an outcome-based design change.

`experiments/frontier_v2_omnisafe.py` implements the pooled PointGoal and
PointButton PPO-Lagrangian/CPO references and is bound to runner digest
`88b7c87a59c2cd6056dc3931dd8cb4c0d65a535463fc21bb921eb41c38e850dc`.
An initial four-world wrapper failed after 13,500 steps because MuJoCo had to
allocate a new model while four task worlds remained resident. The repaired
wrapper preserves the frozen episode-level round robin and seeds but keeps only
one world alive, closes and collects it before a task switch, and closes the
training wrapper before checkpoint evaluation. A real PointGoal
PPO-Lagrangian smoke then completed 50,000 steps, wrote checkpoint SHA-256
`98405fa0c1093c90a15cca8609ecb2c90d698a4bf910ceeb59b29394c60d7594`,
and reproduced its calibration mean score 0.3585 and mean cost 9.75 exactly
after reload. This validates the execution path; it is not a full-seed result.

`experiments/robust_test_subset_baseline.py` implements a separate
RPOSST-inspired comparison in the task-composition layer. It greedily selects a
task subset and uses projected subgradient optimization to fit simplex weights
that minimize the worst absolute full-test score error across tuning policies
and frozen target task distributions. This is not the original k-of-N RPOSST
algorithm and does not inherit its theorem. In an initial Taxi development
matrix with four tasks, three policies, and 50 episodes per task-policy pair,
the two-task subset reproduced the full equal-task policy scores with worst
absolute error 0.00216. The test is useful enough to retain, but broader-domain
calibration and comparisons with uniform and random subsets are still required.

## External benchmark implementation

The development manifest in `experiments/frontier_v2_external_design.py`
contains nine genuinely different domain families from four independently
maintained codebases:

- Gymnasium 1.3.0: FrozenLake, CliffWalking, and Taxi;
- OR-Gym 0.5.0: online knapsack and inventory management;
- Safety-Gymnasium 1.2.0: PointGoal and PointButton; and
- MiniGrid 3.1.0: DynamicObstacles and LavaCrossing.

Each domain has four disjoint development, four calibration, and four declared
confirmation tasks, giving 36 tasks per split. The current manifest hashes are
`6de94c6456eccff522e9f9f359d589d10280f551a9616920f17746652a1c235e`
for development,
`da9faca59d0e1a59e8d98e03d99cdd86b698c5ac618b574492e110d71a2475c2`
for calibration, and
`6538bfd9910eae99be7e692e6d00ba67afbcb1dae68b1eb7ace146fc7aa885b2`
for confirmation. These are development hashes and are not yet registration
locks.

The source audit verifies clean checkouts at Gymnasium commit
`53bf3e9a884783eb72ad3fc8b15780914c97c3e1`, OR-Gym commit
`0b18d16e569e2db70e83f09e867b53bdb4b87298`, Safety-Gymnasium commit
`98231340a4c5b223c8d111fa9597d81836ce09b4`, and MiniGrid commit
`90928729376741a41222a257911343b97103b548`. The complete import-time
dependency stack is explicitly versioned, including dependencies that the
upstream OR-Gym and Safety-Gymnasium packages import eagerly.

Every episode is transformed by a frozen domain rule to a score in [0, 1], so
the paired candidate-minus-fallback difference is always in [-1, 1]. The two
MiniGrid suites operate on fully observable compact images, while both Safety
suites use flattened lidar and proprioceptive observations; four domains
therefore exceed the prespecified 32-coordinate high-dimensional threshold.

A provenance-bound end-to-end rehearsal is current for all 36
development and all 36 calibration tasks. Each task ran the complete
three-policy library for one episode and then repeated the run exactly. Each
split therefore contains 108 episode rows. Both passed task-hash,
whole-manifest-hash, clean source-commit, dependency-lock, canonical seed-block,
common-random-number, complete outcome-schema, derived-summary, score-bound,
deterministic-replay, and runtime-ledger checks. The current development and
calibration runtime sums are 466.74 and 530.76 seconds, respectively. Every artifact is
bound to its outcome-implementation digest, so a policy or adapter code change
makes the artifact fail the current audit. A portability repair now passes the
exact audited source path to Git's `safe.directory` setting for each read-only
provenance command. That repair changed the current digest to
`2c60837f7c36d8b886de5152de206db436a5a5d966b5279587cec121676ee5cf`.
Both splits passed under that digest. A later statistically sized Safety run
found that repeated resets could exhaust native MuJoCo model memory. The
episode-lifetime repair changed the current digest to
`fb8e2a110caa58938701342e2ed5be337e81a3ed61a59c87fd68a9de2a688a73`;
the `2c60837...` and older `5ea81a...` rehearsals are now intentionally stale
and must be regenerated before registration.
The split manifest hashes remained
`6de94c6456eccff522e9f9f359d589d10280f551a9616920f17746652a1c235e`
and `da9faca59d0e1a59e8d98e03d99cdd86b698c5ac618b574492e110d71a2475c2`.

This is full adapter coverage, not a statistically informative policy
comparison and not evidence that the scripted policy libraries are competitive.
`experiments/frontier_v2_full_rehearsal.py` will declare a split complete only
after all 36 exact task artifacts pass the strict audit and unconditionally
refuses confirmation. Statistically sized development and calibration runs,
score-bound stress tests beyond observed trajectories, and trained
DQN/PPO/safe-RL references remain gates before registration.

The rehearsal also triggered a policy-library repair before any confirmation
lock. The original CliffWalking fallback used shifted dynamics and all three
policies were identical on every state. The repaired fallback is planned under
nominal nonslippery dynamics, while candidates use the observed task dynamics;
the risk-averse candidate additionally pays a frozen near-cliff state penalty.
In 20 paired development/calibration episodes per policy, its mean score effect
was mildly negative on deterministic tasks (-0.0059 to -0.0125) and strongly
positive on slippery tasks (0.5725 to 0.6979). The fast shifted-model candidate
gained 0.6315 to 0.7360 on slippery tasks and tied fallback on deterministic
tasks. FrozenLake now uses the same nominal-versus-shifted distinction: the
hazard-averse candidate tied fallback on deterministic tasks and gained 0.2391
to 0.5563 on slippery calibration tasks and 0.3305 to 0.3960 on slippery
development tasks. These are development diagnostics, not confirmation
effects.

Taxi candidates disagree with fallback on 8--9% of tabular states, but a
20-episode paired diagnostic found only rare trajectory changes and effects
near zero. Taxi is retained as a difficult/null-routing domain; its weakness is
not hidden by tuning an artificial score effect. Proposal freezing may exclude
zero-contrast task-policy pairs using development and calibration data only.

The current machine has six CPU cores, 15.8 GB RAM, and a 4 GB GTX 1650, so
scripted-policy development can run locally, but training or evaluating the
high-dimensional reference policies may require a separately recorded compute
environment.

## Empirical questions

The v2 confirmation should distinguish four claims:

1. **Validity:** Does the empirical null-family false-deployment rate agree
   with the theorem under synthetic and simulator nulls?
2. **Efficiency:** How many paired episodes are saved relative to the fixed
   nine-batch and fixed-sample gates at matched familywise level?
3. **Utility:** What equal-domain improvement remains after the stronger
   deployment margin is enforced?
4. **Breadth:** Does the result remain positive when domains, rather than only
   tasks within domains, are resampled?

The primary estimand remains an equal-domain router-minus-fallback effect, but
the study must also report total pilot cost, acceptance precision and recall
against separate final outcomes, harmful accepted routes, unresolved routes,
and per-domain effects. The bounded primary score is immutable after
registration; within-domain task resampling, domain resampling, the task
sign-flip diagnostic, and leave-one-domain-out analysis are prespecified
sensitivities.

Synthetic comparisons use deterministic task-specific random streams. The
same trial seed therefore gives every allocation method the same latent stream
for each task even when the methods visit tasks in different orders. Efficiency
contrasts can be analyzed as paired trial-level differences.

## Development gates before preregistration

These gates determine whether the design is mature enough to register. They do
not determine whether final results are published.

`experiments/frontier_v2_readiness.py` now enforces these execution gates in a
single machine-readable audit. It verifies all seven clean source locks, both
36-task portable rehearsals, separate 20-episode-per-policy development and
calibration suites with exact replay, all 12 learned baseline manifests and
physical checkpoint schedules, calibration selection, selected-checkpoint
replay, all six nonlearned reference manifests, and the current-hash statistical
calibration suite. The current audit passes the complete 6/6 nonlearned gate
and 4/12 learned gate. The new-digest primary, predictable, Bentkus, paired,
close-comparator, proof-certificate, and resolution-bound reruns; the
calibration-sized suite; the router lock; and eight missing learned references
remain explicit failures until their audited jobs complete. The development
20-episode suite completed all 36 tasks and 2,160 episode rows under the current
outcome hash, and its strict audit passed the task-coverage, episode-count,
canonical-seed, common-random-number, replay, and score-bound checks.
It continues to state `confirmation_execution_authorized: false` even after
all readiness checks pass; preregistration remains a separate required action.

- A machine-checked proof appendix matches the implemented e-process formula.
- Unit tests cover thresholds, bounds, alpha weights, terminal decisions, and
  allocation invariance.
- At least 10,000 synthetic null families show no detectable inflation beyond
  Monte Carlo uncertainty.
- Development-only simulations compare power and episode cost across all
  familywise-valid baselines.
- Every external adapter passes seed determinism, common-random-number pairing,
  score-bound, and clean-up tests.
- A compute rehearsal completes without opening any confirmation outcome.
- The complete task family, candidate policies, score bounds, margins, alpha
  weights, allocation rule, maximum budget, seeds, inference, and reporting
  commitment are hashed and publicly registered.

## Initial development diagnostics

These numbers are implementation diagnostics, not confirmation results. They
were produced under the immediately preceding statistical digest
`9c71d848c92cc2a7b12103fed3881cdc7bd5d02d27b60b0593e4e45b109d4192`
and are retained only as development references; the current Bentkus-inclusive
digest must pass complete reruns before registration. With
23 bounded synthetic tasks, familywise alpha 0.05, a maximum of 200
observations per task, and a shared budget of 1,150 observations, the completed
pre-Bentkus 10,000-family global-null gate produced 52 false-deployment
families for primary certified betting: rate 0.0052, Wilson 95% interval
[0.00397, 0.00681]. In separate 10,000-family runs, the predictable plug-in
mixture produced 171 false-deployment families under both uniform and
resolution allocation: rate 0.0171, interval [0.01474, 0.01983]. Every upper
bound is below the registered 0.05 target. This checks the implementation under
adaptive interleaving; the theorem, not simulation, supplies the validity
claim.

In 300 paired-stream mixed-effect families at the same budget, uniform betting
accepted 11.1% of truly positive tasks and produced mean equal-task expected
improvement 0.0187. Resolution allocation accepted 27.8% and produced 0.0382.
The paired improvement difference was 0.0194 with a normal 95% interval
[0.0173, 0.0215], and the paired positive-task acceptance-rate difference was
0.1667 [0.1518, 0.1816]. Neither betting strategy falsely accepted a null task
in that run.

At the same maximum budget, fixed-sample Hoeffding with either Bonferroni or
Holm accepted 15.2% of positive tasks, the uniform Hoeffding mixture accepted
8.3%, and alpha-spending racing accepted 1.5%. The fixed sign methods accepted
23.6%, but they use a distinct independent-sign null and had a 0.0067 empirical
false-deployment rate in these 300 families. The results establish that the
active betting heuristic can materially improve utility under a binding budget,
while the conservative racing baseline supplies a transparent reference.

The frozen planning gaps were deliberately false in the primary null stress
test, illustrating that allocation misspecification does not alter the alpha
guarantee. In the pre-Bentkus 300 mixed-effect families, certified allocation
accepted 38.8% of positive tasks and produced mean equal-task improvement
0.0522. Relative to uniform allocation, the paired acceptance-rate difference
was 0.2762 [0.2638, 0.2886] and the paired improvement difference was 0.0334
[0.0317, 0.0352]. It made no false acceptance in that run. Predictable
resolution accepted 25.5% of positive tasks with improvement 0.0344, while
predictable uniform accepted 13.7% with improvement 0.0221; each had a 0.0133
false-deployment-family rate in this small non-null diagnostic. Fixed betting
therefore remains primary, and predictable betting remains a prespecified
comparator rather than a data-selected replacement.

The binding-budget power diagnostic does not by itself exercise the
untruncated sample-complexity statement: at its 200-observation task cap, 22 of
23 mixed-scenario quotas are clipped. A separate pre-Bentkus check therefore
used 23 two-point tasks separated from the deployment margin by their exact
planning gaps, an untruncated maximum task quota of 5,127, and the full summed
quota budget of 66,810. All 1,000 families respected that deterministic budget.
Three contained an unresolved task and none resolved in the wrong direction,
for combined failure 0.003 with Wilson 95% interval [0.00102, 0.00878]. Mean
realized cost was 18,554 paired observations. This checks the implemented
gap-dependent bound in a nonbinding regime; the mathematical argument, rather
than the simulation rate, remains the guarantee.

The current proof certificate separately checks all seven positive endpoint
betting factors, mixture weights summing to one, 36 task-level alpha weights
summing to 0.05, and exhaustive stopped null expectations through horizon
eight. Fixed betting enumerated 510 paths with maximum expected stopped
acceptance e-value 1.0; predictable betting enumerated the same paths with
maximum 1.0 up to floating-point precision. The certificate is bound to the
same statistical implementation digest as the null and power artifacts.

## Evidential firewall

V1 is complete and remains unchanged. Its pilot and final outcomes may motivate
v2 but cannot be used as v2 confirmation data. V2 development tasks may reuse
environment implementations, but confirmation task parameters and seeds must
be disjoint. All positive, null, and negative v2 outcomes will be reported.
