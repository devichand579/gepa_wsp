# Strict parameter-edit experiment (primary; user confirmed)

The user explicitly selected strict edits: exclude internal network runs and
analytical fitting. Earlier Python-program experiments do not satisfy this
condition and are exploratory only. No final-test inference has been run.

## Why the execution interface changed

The NumPy-only Python namespace still allowed the model to embed feedback images,
implement network layers itself, compute their activations, and fit prototype or
covariance-based classifiers. Disabling autograd and hiding the dataset files
does not prevent these algorithms. Keyword scans for named optimizers did not
establish compliance. Code inspection exposed the issue before final testing.

Examples in the old artifacts include `feats`/convolution/pooling helpers and
`H = maximum(X @ W1.T + b1, 0)` followed by covariance solves. These may be useful
gradient-free construction programs, but they are not evidence for the strict
method. Preserve every transcript, checkpoint and cost; do not pool their scores
with primary strict results.

## Bounded declarative parameter operations

The LLM now returns JSON, not Python. The interpreter executes only:

- parameter block assignment, sparse scatter assignment, addition, and scaling;
- copying/reshaping/transposing/shifting existing parameter blocks;
- centering and L2-normalizing parameter blocks;
- coordinate formulas over tensor indices and numerical constants, using a small
  fixed set of elementwise mathematical functions.

No general loops, functions, array reductions over examples, matrix solves,
matrix multiplication, model/layer calls, regression, optimization routines,
dataset I/O, file paths or imports are available. Coordinate formulas cannot read
weights or input examples. The only array sources in the interpreter are the
parameter dictionary and numeric constants selected by the LLM. This is a
restricted weight-programming space, not a claim to unconstrained Python editing.

Programs have at most 256 operations and 180K serialized characters. Formula
length/node/constant/power limits, 20 CPU seconds, 2 GiB address-space and numerical
checks bound execution. They run in a minimal no-network namespace with no
dataset, credential, project or host-process access. The trusted host also checks
tensor names, shapes, finite values and module write boundaries independently.
Only the fixed harness executes the target network.

## Fixed tasks and comparison

Keep the original tasks, target architectures, data manifests and seeds 0/1/2:
MNIST 101,770-parameter MLP; Fashion-MNIST 20,490-parameter CNN. Reuse the existing
Adam baselines: their architecture, data and initial weights have not changed.
No Adam weights, outputs or gradients enter LLM context or construction.

Both one-shot construction and GEPA use the same new interface, initial values,
task information, example packet, model and decoding settings. GEPA starts from
that exact construction checkpoint and receives up to forty actual mutation API
calls per seed, including invalid responses. Exact cached-response replays are
not additional calls. No generated submission is manually repaired. Invalid operations are rejected individually; valid operations in the same submission still execute and partial validity is reported. A submission with no valid operations fails.

DeepSeek V4.1 Flash (`accounts/fireworks/models/deepseek-v4p1-flash`), temperature 0.6, maximum 65,536 output tokens including reasoning, with an explicit 8,192-token thinking budget. Plain JSON output: the task/system messages specify the contract and the parser validates it. Schema and tool-call pilots produced malformed payloads and are excluded from the primary runs. The user-authorized alternative replaces MiniMax for all clean primary runs. Plain-JSON bounded-thinking pilots completed both construction submissions; the MNIST submission applied all 114 operations and Fashion applied 9 of 19. This choice concerns submission completion and executability, not final-test scores. A shape-only readout-row syntax example was added after diagnosing malformed indices, without supplying task-specific weights. Decoding and prompts are frozen before primary runs and final-test inference.

Use official GEPA 0.1.4 Pareto selection, a fixed repeating schedule of one joint all-layer edit followed by each individual layer, no crossover, paired
128-example feedback batches and the declared composite score
`0.9*correct + 0.1*p_true`. Selection examples/labels never enter reflection.
The custom batch evaluator produces traces only for feedback batches. Coordinated edits prevent a dead all-zero multilayer network from becoming unreachable through single-layer improvements. Joint programs execute once on the full parent checkpoint; each selected component points to the resulting checkpoint. The fixed network forward is unchanged.

Initial one-shot sees ten illustrative labeled images. GEPA gets additional
examples/diagnostics. Report this exposure difference; do not attribute a gain
solely to search rather than more evidence. A matched-example one-shot control,
if run, must receive only the raw training examples seen by the search, no
selection/test material or optimized weights, and be selected before final tests.

## Budget and evaluation

The whole-project request ceiling already proposed was 650 and the spend ceiling
$20. Keep the stricter $10 current spend cap, inclusive of all earlier pilots and
exploratory calls; extend the request ledger to 650 for the clean replacement
runs. The per-seed forty-mutation cap is unchanged. Record the superseded first
batch's 300-request planning limit rather than silently claiming it was retained.

Freeze the new interpreter, prompt and adapter hashes after integration pilots,
before primary strict runs. Final-test inference remains locked until all six
strict searches and corresponding baselines finish. Preserve errors and negative
results. Three seeds provide exploratory evidence; do not promise positive gains.

## Checks

`check_tensor_dsl.py`: permitted coordinate construction and copies, normalization,
immutability, module bounds, forbidden Python/file/function access, numerical limits.
`check_strict_integration.py`: real GEPA acceptance and full-selection path without
selection trajectories; explicit fitting/Python/file operations are rejected.
`audit.py`: recorded-request provenance checks and credential artifact scans.
`data/duplicate_audit.json`: zero exact cross-split image duplicates in the used
feedback, selection and official test partitions, for all six task/seed pairs.
