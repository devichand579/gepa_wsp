# Current primary protocol: feedback-aware weight-update programs

The user's latest instruction supersedes strict JSON-only edits. Current weights,
intermediate forward inputs/outputs, activations, predictions and permitted
training examples may guide the program. NumPy/math imports are allowed; no unrelated modules are exposed. NumPy statistics, correlations,
covariance, linear algebra and closed-form fitting are explicitly allowed.
This is LLM-generated gradient-free weight-update programming. It is not a claim
that the LLM directly types every fitted coefficient without numerical algorithms.
Autograd, backpropagation, numerical gradient estimation, conventional optimizers
and iterative training/search loops remain excluded. Review generated programs
before making this mechanism claim; NumPy-only execution alone is not a proof
that no hand-written optimizer was implemented.

## Data and comparison

Reuse the same architectures, splits, initializations and seeds0,1,2:
MNIST101770-parameter MLP; Fashion-MNIST20490-parameter CNN; SST-2 83266-parameter
transformer. Image feedback pools contain2000 examples; text4000. Each selection
pool has1000 disjoint examples. Image final tests are official10000-example test
splits; text uses the official872-example labeled validation split as final test.
Exact raw/encoded cross-split input-duplicate audits are saved with the data.
Existing Adam baselines use these same data and random initial weights.

Both one-shot construction and GEPA get identical initial access to the ENTIRE
permitted training pool and its initial forward outputs, plus an intermediate
trace for128 examples. This is a stronger, feedback-assisted one-shot baseline.
GEPA starts from that exact construction checkpoint and gets up to30 mutation API
calls per seed. All modules may be edited in each proposal. Invalid submissions
and provider failures count; cached-response replays do not add calls. Additional
feedback passes and proposals are part of the treatment, while accessible training
data is the same across the LLM conditions. Adam is a conventional separately
trained baseline; no Adam weights/gradients enter the writer context.

GEPA0.1.4, Pareto selection, all-component mutation, no crossover, paired128-example
feedback batches. Score0.9*correct+0.1*p_true. Full selection evaluations use1000
examples. Selection examples/labels/trajectories and final-test material never
enter edit code or reflection. Selection does affect candidate choice, so final
tests remain sealed until all nine rich searches are frozen.

## Rich interface

Each update receives edit(w,f). w contains every full-precision current parameter.
f contains only permitted feedback:
- x,y: the entire training/feedback pool;
- features,logits,probs: full-pool forward outputs under the current parent;
- batch.ids,batch.x,batch.y: the paired reflection batch;
- layer.<module>.inputN/output: complete module inputs/outputs for that batch,
  captured by hooks during its real forward. For example, an MLP exposes both
  linear preactivations and the hidden ReLU input to fc2; CNN module inputs also
  expose the max-pooled activations. Transformer submodule traces expose norms,
  qkv, projections, feed-forward activations, and block outputs.

Activations are those of the parent before the update. Programs may compute
further statistics or array calculations from feedback, but no private evaluator
or dataset files exist inside the sandbox. All arrays are available to the update
program; the LLM prompt contains their schema, task information, full feedback
metrics, per-class feature summaries, intermediate-layer statistics and illustrative
mistakes. Small/full parameter tensors are also shown directly: all CNN weights,
all non-embedding transformer weights, and MLP readout weights. Larger matrices
are summarized with selected rows. Literal prompt values are rounded for size;
program arrays keep full precision. A program may return up to24000 characters of
chosen diagnostics for later reflection, allowing inspection of additional weight
or activation blocks within the same proposal budget.

Programs execute in a minimal isolated namespace exposing Python/NumPy, current
weights and this feedback snapshot, without project paths, credentials, network,
selection/test data or the model object. Numeric output checks enforce names,
shapes, finite float32, and absolute values<=1000. Two seeded executions must
produce identical numeric weights; both executions and wall time are recorded.
No generated program executes during selection or deployment: execute the update
once at proposal construction, save numeric weights, then evaluate the fixed
forward. Selection cannot cause a program to refit on selection examples.

Full activation snapshots are ephemeral to avoid retaining tens of gigabytes.
Save their hashes, complete array schema, parent checkpoint, batch indices,
immutable data manifests and program source so snapshots can be regenerated.
Record additional full-feedback forward example counts separately from GEPA's
metric-call count. Record generated-program execution time; its array calculations
are not falsely counted as no computation or as only the harness forward budget.

## Model, budget and provenance

DeepSeek V4.1 Flash, temperature0.6, thinking budget8192, max32768 total output tokens,
plain Python final output. One call per construction; thirty mutations per seed.
The reduced per-condition budget keeps the replacement suite within the original
650-request/$20 WHOLE-PROJECT ceilings, inclusive of every archived run and pilot.
No fresh spending allowance is assumed. Provider failures retain their request IDs
and count as calls. Unknown billing fails closed, then may be conservatively
settled at the full reserved upper bound by the audited recovery controller.

Preserve earlier Python exploratory artifacts and strict JSON partial runs as
separate protocol variants. They are not pooled with the new primary condition.
The old strict controllers are retired and in-flight calls were allowed to finish.
The first pilots emitted harmless NumPy imports, which the initial worker rejected. Allow NumPy/math imports and replay identical saved responses on byte-identical feedback snapshots without another API call. Preserve the original errors and replay provenance separately. Primary runs start fresh; pilot weights are not transferred. Pilot changes concern interface executability and compact float serialization;
freeze rich sources/configuration before primary runs. Report invalid code,
provider failures, random fallbacks and all seeds. Three seeds remain exploratory.
A greedy-reflection control is still needed to attribute gains specifically to
Pareto selection. Public-data pretraining contamination cannot be ruled out.

## Array-helper compatibility amendment

The first primary Fashion construction used NumPy sliding_window_view through np.lib.stride_tricks. The inherited namespace check rejected lib, a public numerical helper namespace. Remove that blanket lib rejection and allow imports from numpy.lib.stride_tricks; retain file/network/private-attribute protections. Every subsequent worker report records its actual source hash. Original primary errors and weights are preserved; no primary submission is edited or replaced. That specific generated PCA program also requested16 eigenvectors from a9-dimensional covariance and is not assumed valid after the compatibility fix. This amendment restores the stated public-NumPy capability without changing prompts, data, score, call budget or target model. Prior worker/freeze snapshots are retained under runtime/rich_before_array_helper_fix.
