# Reflective weight optimization: one-day experiment plan

Status: historical original plan. Execution is underway. The user confirmed strict weight edits; see STRICT_PROTOCOL.md for the primary amended protocol and runtime/STATUS.json for progress.

## Question and scope

Can an LLM construct useful neural weights and improve them through GEPA's
reflection and candidate selection, without gradients or conventional numerical
weight optimizers? Compare against one-shot LLM construction and supervised Adam.
This is learning/training by reflective search, not an absence of learning.

Deliver a reproducible exploratory blog within 12–24 hours of execution starting.
Commit to MLP, CNN, and transformer experiments; include image diffusion as an
exploratory fourth experiment. Report negative results and incomplete runs.
Do not change tasks after seeing final test performance.

## Observed machine state

- Two NVIDIA RTX 4500 Ada Generation GPUs, about 24 GiB each.
- At inspection GPU 0 had about 24.1 GB free; GPU 1 about 3.85 GB free.
- GPU 1 had an existing VLLM::EngineCore process using about 20.25 GB.
- Leave existing processes and environments untouched.
- Existing amda environment has torch, torchvision, sklearn, datasets, openai,
  and numpy. GEPA and plotting dependencies need an isolated project environment.
- First test CUDA and peak memory in that environment. Assign small classifier
  workers to GPU 1 only if memory and utilization allow; reserve GPU 0 for the
  diffusion worker. Use bounded queues and CPU fallback rather than interrupting
  existing workloads. API-hosted reflection does not occupy the local GPUs.

## Reflection model

Primary: MiniMax M3.1 Flash via Fireworks,
`accounts/fireworks/models/minimax-m3p1-flash`.
Official page: https://fireworks.ai/models/fireworks/minimax-m3p1-flash

DeepSeek V4.1 Flash is the user-authorized alternative, but its exact served ID
still needs verification. Do not guess the ID or substitute another model.
After plan review, verify credentials with a minimal request, model response,
code formatting, usage fields, and latency. Pin model ID and sampling parameters.
Use one model for all primary comparisons. If an endpoint fails, resume it or
label a replacement run separately; do not silently mix reflection models.

The supplied credential must never appear in source, command arguments, logs,
figures, saved prompts, or the blog. Resolve it through a private runtime secret.
Edit subprocesses receive no credential or network access.

Proposed entire-project spend cap: USD 20. Implement an application-side usage
ledger with conservative reservations for concurrent calls and retries; stop
before the cap. Record token usage including billed reasoning where available.
Missing usage or unknown pricing must stop paid dispatch rather than assume zero.
No paid calls during this planning step.

## Tasks and networks

1. MNIST, 28x28, ten-class classification.
   MLP 784 -> 128 -> 10, ReLU, 101,770 parameters including biases.
2. Fashion-MNIST, 28x28, ten-class classification.
   Conv(1,16,3,pad=1), ReLU, MaxPool(2), Conv(16,32,3,pad=1),
   ReLU, MaxPool(2), Flatten, Linear(1568,10).
   20,490 parameters including biases.
3. SST-2 sentence sentiment classification.
   Word-level vocabulary built only from feedback data, approximately 2,000
   tokens; width 32, 2 encoder blocks, 2 heads, FFN width 64, max length 64,
   masked mean pooling, binary head. Approximately 80–100K parameters; report
   exact implemented count and padding/masking behavior before runs.
   No pretrained tokenizer embeddings or pretrained target-model weights.
4. MNIST resized to 16x16, unconditional image diffusion.
   Fixed forward noising schedule and fixed DDPM reverse sampler. Small
   convolutional denoiser with timestep conditioning, target 50–150K parameters.
   Freeze exact architecture, noise schedule, and sampling steps before main runs.
   This experiment is exploratory; denoising improvement is not generation success.

## Data and feedback

- Use immutable split manifests and seeds 0, 1, 2. Normalize with fixed constants
  or feedback-split statistics only.
- For image classification, initial budget: 2,000 class-balanced feedback
  examples and 1,000 selection examples from the official training partition.
  Evaluate final selected checkpoints on the official test partition.
- For SST-2, use 4,000 feedback examples and 1,000 disjoint selection examples
  from its training partition. Reserve labeled official validation examples for
  final testing because official test labels are not generally provided.
- Diffusion: 2,000 feedback images, 1,000 disjoint selection images, official
  test images for final distribution comparison. Noise and timestep draws for
  candidate comparisons are paired and reproducible. Final evaluation uses fresh
  noise seeds and generated samples.
- Every LLM method gets the same initial labeled-example packet, architecture,
  parameter schema, task description, and initialization. GEPA receives additional
  feedback as an explicit part of the method. Report examples actually exposed;
  do not claim identical information or compute across methods.
- Show bounded feedback batches, errors, logits, confusion matrices, parameter
  statistics, and selected activations. Training examples may be shown; selection
  inputs/labels and final test contents are never exposed to reflection.
- Per-instance selection scores support GEPA selection. Keep the final test
  strictly outside candidate selection and protocol development.

## Methods

A. One-shot LLM: one code submission constructs weights from fixed initialization.
   It can edit all tensors but cannot evaluate candidates or invoke fitting code.
   A syntax/execution failure counts; do not grant free corrective submissions.
B. LLM + GEPA: initialize with A's exact checkpoint, then allow 40 mutation
   proposals. This directly measures added value beyond construction. A failed
   construction leaves the shared initial checkpoint and is reported.
C. Adam: same architecture, data pool, and seed initialization. Use three fixed
   learning rates (1e-4, 1e-3, 1e-2) and predeclared epoch limits; choose learning
   rate and checkpoint only by selection performance. Log all tuning cost.
   Adam is a separate baseline, never used to initialize or repair A/B.
D. Optional attribution ablation: greedy reflective editing on CNN, same
   initialization, feedback machinery, and mutation budget. Required for any
   claim that Pareto selection specifically causes improvement.

## GEPA representation and search

Use the official GEPA implementation through a custom adapter; pin its version.
Source: https://github.com/gepa-ai/gepa

A candidate consists of per-module tensor-construction programs materialized into
a complete immutable checkpoint. Programs reconstruct the exact candidate from
fixed seed tensors; dependencies between module edits must be explicit and hashed.
All baseline and search inference uses the same unchanged PyTorch forward code.
Thus inference-time decisions come from network weights, not edit-program logic.

Modules: MLP layers, CNN layers, transformer embeddings/attention/FFN/head blocks,
and denoiser convolution/time-conditioning blocks. Fix the module-selection
policy before main runs, initially round-robin. Full-network construction is
permitted in A. Search mutates a module and materializes the complete child.
Disable crossover for the initial suite: untested composition of weight edits
would add another experimental variable.

Reflection writes short structured tensor operations rather than dumps of
100,000 numbers. All tensors remain accessible for edits. Report changed-weight
counts, update norms, effective ranks where useful, and frozen tensors.

The edit worker is isolated, time-limited, has no datasets/evaluator, network,
secrets, or arbitrary filesystem access, and only returns tensors. Reject NaNs,
shape changes, altered architectures, inference-time code injection, gradients,
finite-difference estimation, optimizer calls, closed-form fitting routines, and
internal trial-and-error loops. Deterministic loops constructing tensors are fine.
Every attempted proposal, rejection, and evaluation is counted and logged.

Use paired feedback batches to compare parent and child. Record accuracy and
cross-entropy, with a predeclared scalar score and tie policy. Selection uses
per-example correctness for classification. For diffusion use negative per-image
noise MSE averaged across fixed noise/timestep draws. Repeated selection is
adaptive validation, not independent evidence of generalization.

## Evaluation and claims

- Classification: final accuracy, cross-entropy, per-class scores, examples.
- Diffusion: held-out denoising MSE plus a fixed, declared distribution metric
  (e.g. pixel-space sliced Wasserstein distance) and uncurated generated grids.
  Label metric limitations; do not call pixel distance FID or evidence of realism.
- Report all seeds individually plus means and variability. Three seeds are
  exploratory evidence, not a definitive statistical study.
- Plot selection curves during search; final test scores are computed after
  candidate selection, not used for steering. If test learning curves are shown,
  compute them post hoc after freezing all decisions.
- Record wall time, API tokens/cost, candidate evaluations, examples evaluated,
  Adam training/tuning steps, peak GPU memory, and external workload contention.
- Do not equate Adam steps to GEPA proposals. No efficiency claim without a
  stated resource comparison. Modest gains and failures are publishable outcomes.

## Run budget and schedule

- Pilot: all four tasks, seed 0, at most five mutation proposals each; checks
  infrastructure and Adam learnability, never final test outcomes.
- Main: 4 tasks x 3 seeds x (1 construction + 40 mutations) = 492 LLM calls.
  Plus pilots, bounded transport retries, and optional greedy control. Hard cap:
  650 paid requests total AND USD 20, whichever is reached first. Count failed
  paid requests. Output and context token limits are pinned after pilot.
- Pilot changes are logged; freeze configurations before main runs. Do not
  selectively lengthen budgets for promising tasks or discard bad seeds.
- Hours 0–3: isolated environment, adapter, model/data definitions, enforcement
  checks, GPU smoke checks, usage ledger, Adam sanity runs.
- Hours 3–6: API and end-to-end pilots; freeze main protocol.
- Hours 6–16: two GPU queues, main classification and diffusion experiments.
- Hours 16–20: finish repetitions and any prebudgeted ablation; freeze selection.
- Hours 20–24: final tests, plots, reproducibility pass, blog draft.
- If behind schedule, finish the three classification tasks and report diffusion
  as a pilot. Do not shrink architectures or relabel tasks silently.

## Deliverables

Runnable source/configurations, dependency lock, split manifests, sanitized
API transcripts, mutation programs, immutable checkpoints and hashes, complete
metrics/usage records, figures, and BLOG.md with limitations and negative results.
Write a local draft first; publishing is a separate action.

## Items to resolve before execution

Prompt design and source comparisons are documented in PROMPT_PROTOCOL.md.
Freeze the system/task/reflection templates after pilot. Check reasoning/output
token headroom; a 16K output cap may be too restrictive for construction.

Confirm the proposed USD 20 cap and whether the intended fourth task should be
committed or exploratory during plan review. Exact model sizes, environment,
adapter semantics, code-execution isolation, output limits, and GPU headroom
are implementation checks before dispatching main runs.
