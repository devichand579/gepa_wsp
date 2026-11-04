Current primary protocol: see **CURRENT_EXPERIMENT.md** and **RICH_PROTOCOL.md**. The user superseded strict JSON edits with feedback-aware NumPy programs, permitting statistics and closed-form fitting. Active artifacts are under `rich_runs/`; earlier protocols are preserved separately.

# GEPA weight-space experiments

Exploratory gradient-free weight construction and refinement on MNIST (101,770
parameter MLP) and Fashion-MNIST (20,490 parameter CNN), compared with one-shot
LLM construction and independently trained Adam. Transformer and diffusion are
later phases; they are not part of the first batch's completed-result claims.

The primary protocol is STRICT_PROTOCOL.md. EXPERIMENT_PLAN.md and PROMPT_PROTOCOL.md preserve the original plan and prompting analysis. Actual configuration is saved with each run. The clean replacement suite has a $10 spend cap and 650-request ceiling including all pilots and exploratory work, within the original $20 whole-project ceiling. The superseded first-batch request ceiling was 300.

## Information boundary

- Feedback: 2,000 balanced training examples per task/seed; construction sees ten
  illustrative examples and search sees bounded diagnostics from 128-example
  feedback minibatches. This is not identical example exposure to Adam.
- Selection: 1,000 disjoint training examples, evaluator-only. GEPA gets scores
  for selection, but reflection never receives their inputs, labels, or traces.
- Final test: official test split, evaluator-only after all six first-batch model
  selections are frozen. `finalize.py` refuses incomplete suites.
- Generated declarative JSON tensor operations run in a minimal bubblewrap namespace containing only
  Python, NumPy, the fixed worker, candidate weights, and an output directory.
  No project, datasets, API credentials, network, or host process namespace.
- Candidate edit programs are parameter construction tools only. Deployment is
  the unchanged neural forward pass over the resulting tensor checkpoint.

## Reproduction

The project venv uses existing torch/CUDA libraries through system-site-packages.
Exact packages and runtime versions are recorded separately. A clean independent
environment needs compatible torch, torchvision, numpy, scipy, gepa and matplotlib.

```
.venv/bin/python check_tensor_dsl.py
.venv/bin/python check_strict_integration.py
.venv/bin/python prepare.py mnist
.venv/bin/python prepare.py fashion
.venv/bin/python experiment.py --task mnist --seed 0 --device cuda:0 --phase pilot --method adam
.venv/bin/python strict_experiment.py --task mnist --seed 0 --device cuda:0 --phase strict_pilot --proposals 1
```

Runtime authentication is read from a private file outside the project and never
passed to edit subprocesses. Do not put keys into run configs or shell arguments.

GEPA 0.1.4 provides the evolutionary engine and instance-wise Pareto selection.
Our adapter proposes module tensor programs. Crossover is disabled. The fixed
per-example score is `0.9 * correctness + 0.1 * probability_of_correct_class`:
accuracy is the main component and probability supplies a smooth auxiliary signal.
Reported accuracy remains ordinary argmax classification accuracy.

Adam uses the same model, feedback data and random initialization, batch size
128, three learning rates (1e-4, 1e-3, 1e-2), at most 50 epochs per rate and
patience 10. Count all tuning cost. Selection chooses its checkpoint by the same
composite score; final evaluation reports accuracy and cross entropy.

One-shot and GEPA share the exact construction checkpoint. A failed construction
is reported and leaves the random initialization as the checkpoint. Invalid operations are rejected individually; other valid operations execute, and partial validity is reported. A submission with no valid operations receives a failing score. All submissions count without correction calls. Global service/budget failures stop the experiment and must not be
reported as learning failures.

## Outputs

`runs/{pilot,main}/{task}/seed{n}/`: sanitized complete LLM requests/responses,
construction JSON, checkpoints, GEPA lineage, evaluation records and Adam history.
`runtime/usage.json`: concurrent-safe token/cost ledger. No credentials.
`runs/main/FINAL_RESULTS.json`: produced only by the separate final evaluator.

Three seeds are exploratory. Conditional paired test-example intervals do not
measure uncertainty from the entire training/search procedure. A comparison
against greedy reflection is needed to isolate GEPA's selection contribution.

Earlier Python-program runs are preserved under `runs/exploratory_python/`. Some performed internal network execution or analytical fitting. They are excluded from primary strict results. Primary target-model forward calls happen only inside the trusted evaluator; the tensor interpreter has no data access, model calls, matrix products, or fitting operations.
