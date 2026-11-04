# Implementation amendments

## Split-aware GEPA batch evaluation

The original adapter rejected any request for selection-set trajectories. GEPA
0.1.4's initial seed validation uses capture_traces=False, but its default batch
helper uses capture_traces=True when validating accepted mutations. Consequently
the information-boundary assertion stopped runs at their first accepted mutation.
The assertion fired before selection examples were turned into traces. No
selection examples or labels entered reflection.

Fix: implement batch_evaluate on the adapter, explicitly selecting traces only
for feedback batches. Keep the direct-call assertion and the reflective-dataset
provenance assertion. A new integration test invokes the real GEPA engine,
forces an improved candidate, and verifies the full-selection path returns no
selection trajectories.

This changes the adapter integration, not the task, model, prompts, data,
acceptance rule, candidate selection, or proposal budget. The original source,
protocol hash manifest, run configurations, and failure logs are retained.
Resume from existing GEPA states and identical construction checkpoints. Reuse
a prior API response only when its complete saved request exactly matches the
reconstructed request. Otherwise retain the old attempt and record any new
request separately, including its cost. No generated code is manually repaired.

Dispatch is now directly concurrent over three independent seeds per GPU, rather
than pausing serial dispatchers. This is scheduling only. All seed budgets and
data splits remain unchanged. Final evaluation remains locked until all six
search runs and Adam baselines finish.

## Standard Python builtins

The initial restricted builtin list inadvertently omitted divmod, rejecting a
Fashion-MNIST seed-2 construction even though it used ordinary deterministic
index arithmetic. Expand the safe builtin set to ordinary Python functions and
exceptions, retaining the explicit import/file/evaluation/introspection bans,
AST checks, minimal filesystem namespace, network isolation, resource limits,
and tensor constraints. Test divmod and pow alongside NumPy dispatch and the
accepted-candidate GEPA integration.

Archive the entire affected Fashion-MNIST seed-2 search as a harness-invalid run.
Restart it using its exact original API construction response, without manually
changing the generated program. Other seeds resume from saved states. All
discarded-run API costs remain in the global ledger. Keep all original snapshots.

Enforce the per-run mutation limit using actual completed API calls, including
preserved restart attempts. Exact response replays do not cost an additional
proposal. If a changed resume packet requires a new call for an interrupted
iteration, it consumes one of the forty calls rather than creating a free extra
attempt. Report both API proposal count and completed GEPA iteration count.


## Provider interruption and conservative recovery

The suite stopped on a TimeoutError and HTTPError with unknown provider usage. No final-test inference occurred. The two pre-dispatch cost reservations sum to $0.2327385. Charge the entire upper bound for each unknown request (do not invent usage or assume zero billing), retain the error and request ID, then resume within the original $20/650-call whole-project ceilings. Future provider interruptions still fail closed; a separate bounded recovery monitor can settle full reservations and resume after all in-flight calls finish. At most three further recovery rounds are allowed.

Every actual dispatch, including a provider failure, consumes one of the same forty mutation slots. Dispatch metadata now records the ledger ID before network I/O. Cached response replays do not count twice, and a recorded provider failure is not retried in the same slot. The former metadata-only counter missed failed requests. The GEPA iteration safety ceiling now includes twenty slots of checkpoint/replay overhead; the paid mutation cap remains forty, and corresponding metric-call headroom changes only for lifecycle recovery. No prompt, decoding, tensor-language operation, data split, architecture, score or model changed.

Reduce concurrent provider calls to three (one image run per GPU plus one text run) after the interruption. Preserve original sources, frozen configurations and per-run configs under runtime/before_service_recovery and run_config_before_service_recovery.json. Record revised source hashes as a lifecycle amendment, keeping earlier generations and selections intact. Provider failures must be reported separately from invalid tensor edits and learning outcomes.
