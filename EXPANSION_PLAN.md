# Expanded study under an inclusive $50 ceiling

The user authorizes $50 for ALL experiments, including historical pilots, strict
variants, current rich-feedback runs and failed requests. The shared ledger has
this hard cap and a2000-request safety ceiling. Model prices are conservatively
accounted at uncached rates; unknown billing is charged at the entire reservation.
No dedicated paid deployments or new paid providers.

Respect the user's Codex-credit preference: local scripts perform training,
queueing, recovery, auditing and report assembly. Use bounded setup/check-in
turns; do not run a continuous Codex polling loop. A local sleeping worker costs
no Codex inference. Goal remains active while required work is incomplete.

## Frozen follow-up matrix (before final-test access)

1. Finish current DeepSeek rich classification runs: all three tasks,seeds0/1/2,
   thirty proposals. Extract prefix results at budgets0,5,10,30 without extra calls.
2. Continue those SAME trajectories to sixty proposals in separate copied artifact
   roots, preserving the original thirty-budget checkpoints. All tasks and seeds.
   Extract/select budget60 without retrospectively choosing a trajectory by test.
3. Fresh one-shot+ten-proposal classification runs with MiniMax M3.1 Flash and
   Nemotron Lightning3.5 30B A3B, all three tasks and seeds0/1/2. Compare at matched
   budgets0,5,10 using prefixes. Same data, architecture and rich feedback access.
4. Add unconditional MNIST16x16 diffusion: fixed32-step cosine DDPM,56609-parameter
   convolutional epsilon predictor, timestep concatenated as an input channel.
   Compare one-shot, GEPA and separately trained Adam. LLMs: DeepSeek, MiniMax,
   Nemotron; seeds0/1/2; proposal checkpoints0,5,10,20. Match noise/timesteps for
   candidate comparisons; final losses use fresh noise. Publish fixed-seed sample
   grids as well as denoising MSE. Lower MSE alone is not generation success.
5. Audit generated programs, freeze ALL expanded model selections, then unlock
   final tests. Produce a combined budget/model/task report with costs and failures.

Statistical fitting and training-only internal model selection are part of the
feedback-aware weight program and must be identified in the report. They are not
held-out leakage and cannot be credited solely to GEPA's Pareto mechanism.
Conventional gradient/optimizer loops remain excluded from LLM updates; Adam is
separate. No selection/test data enters update code or reflection. Code audit is
needed before claiming optimizer absence.

## Spending allocation (planning envelopes, shared hard cap)

- Original work and current30-call classification suite: up to$20 total.
- Continuations30->60: approximately$12.
- Cheaper model classification comparisons: approximately$7.
- Diffusion comparisons: approximately$9.
- Recovery reserve:$2.

Total$50. Reuse baselines and prefix checkpoints. Keep the shared cap authoritative;
unused allocations can cover another phase, but never exceed$50. Expected total
request count is about1263 including historical work, below the2000 safety cap.
Do not spend the reserve merely because it is available. A budget shortfall must
be reported as incomplete, not silently reduced scope or a completed goal.

## Verified hosted models

- DeepSeek: accounts/fireworks/models/deepseek-v4p1-flash; conservative existing
  ledger rates$0.30/$1.20 per million input/output tokens.
- MiniMax: accounts/fireworks/models/minimax-m3p1-flash; existing verified rates
  $0.15/$0.60. Verify again before the follow-up paid batch.
- Nemotron: accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b, listed by the
  authenticated serverless model endpoint. Official page reports$0.05/$0.20
  input/output, cached input$0.01. Record a capability probe before the batch.
  https://fireworks.ai/models/fireworks/nemotron-lightning-3p5-30b-a3b

Qwen3.5 9B and Qwen3 Coder30B pages currently say serverless is not supported, so
we will not start costly on-demand deployments to use them.

## Implementation status

The current classification queues are live. The inclusive cap is updated.
Diffusion data preparation, denoiser, cosine schedule, paired noise, sampler and
Adam baselines are implemented and checked. Remaining: generic model client and
follow-up orchestration, prefix/continuation verification, diffusion rich-feedback
GEPA adapter, capability probes, all follow-up executions and final report.


## Implementation checkpoint

Shared per-process model/client and pricing registry: expansion_client.py. No primary source file was changed. Capability probes for MiniMax and Nemotron are preserved under expansion_runs/probes/. Classification continuation/model runner: classification_followup.py. Nested budget prefix selection passes a future-candidate exclusion test; completed real-run prefixes are also verified by independently rerunning their numeric checkpoints on selection data, with evidence in runtime/check_real_budget_prefixes.json. Remaining work includes the diffusion GEPA adapter, automated follow-up queue/recovery, running the expanded matrix and final combined evaluation/report.


## Diffusion and queue implementation checkpoint

Diffusion GEPA is implemented in diffusion_experiment.py. Its integration check exercises real GEPA acceptance, verifies denoising outputs against the separately trained baseline evaluator, and confirms selection evaluation never executes update code. All three diffusion Adam baselines are complete. A hash-only audit checks the resized inputs across splits; no final labels, predictions or metrics are used. The local expansion_queue.py schedules27 classification follow-ups and9 diffusion jobs. It verifies required budget prefixes/hashes before freezing expanded selections and has bounded provider cooldown/recovery. The combined final report remains to be implemented.


## Context compatibility amendment (before valid diffusion submissions)

Nemotron rejected the initial diffusion payload:264198 prompt tokens exceeded its262141-token context. Four experimental dispatches and diagnostic probes are preserved and conservatively charged; no valid update program was produced by that condition. Archive it under diffusion_runs/provider_rejected/. Public diagnostic full-output settings work; the issue is context length.

All diffusion models and all FRESH ten-call classification model comparisons now share compact display prompts, bounded to180000 UTF-8 bytes plus the fixed output budget. Full-precision weights and feedback arrays remain completely accessible inside edit(w,f). Large literal parameter tensors are represented by selected blocks and statistics; older verbose numeric reports may be omitted from the prompt when needed.

Add fresh DeepSeek ten-call classification runs with this identical compact representation, alongside MiniMax and Nemotron, for matched model comparisons. Preserve original-representation DeepSeek30->60 trajectories unchanged for the budget study. This yields45 follow-up jobs and roughly1370 total project requests including compatibility failures/probes; the inclusive$50/2000 safety ceilings are unchanged. All choices precede final-test inference.


## Combined evaluator implemented

expansion_finalize.py now verifies the complete45-job selection freeze, source/checkpoint hashes, feedback provenance and selected-program lineages before reading final test data. Flagged selected optimizer-like code requires a recorded statistical-method review; both primary and expanded test evaluation wait for this review gate. The evaluator caches predictions by weights and test-array hashes, compares all declared budget prefixes and Adam, generates fixed-seed diffusion grids, reports conditional paired statistics, and assembles BLOG_EXPANDED.md plus expanded_results/results.json and summary.csv. All final evaluation is local and makes no LLM calls. Integration checks exercise the evaluation workflow on synthetic arrays without reading actual test data. Remaining work is experiment execution and any required mechanism review, followed by completion verification.


## Method-violation repair before final tests

Source review found executed parameter-gradient/Newton updates in DeepSeek MNIST seeds1/2 (including an unselected successful candidate) and the Nemotron diffusion seed0 constructor. Original results and costs remain preserved and are not eligible as gradient-free evidence. Fresh complete replacements use the SAME model, data split, target seed and60/20 proposal budget. This is a method-compliance correction, not selection by final-test performance; no final tests have run. The retained original trials still count toward the inclusive$50 ceiling.

A pre-execution AST dependency screen rejects explicit parameter-gradient/Newton values that flow into returned weights, while preserving closed-form fits, spatial np.gradient on feedback inputs and diagnostic-only statistics. It is conservative and is not a proof against arbitrary disguised algorithms; final selected-lineage source review remains required. All named violations are regression checks. New repair paths and reasons are in runtime/EXPERIMENT_REPAIRS.json. The original primary-only finalizer is superseded so it cannot release invalid pre-repair states; expanded_results is the authoritative final evaluation. Local queue overrides preserve all45 planned jobs and all original prefix budgets.


### Broader source-review amendment before final tests

Review extended beyond exact `grad` identifiers and found additional executed Newton/gradient updates in DeepSeek MNIST seed0, Fashion-MNIST seeds1/2 and Nemotron diffusion seeds1/2. All eight affected trajectories, including unselected executed violations, receive complete fresh replacements at unchanged seeds/models/budgets. Their historical costs remain charged. The method screen now covers gradient suffixes, Hessian/Newton and derivative helpers; human lineage review remains necessary. The whole-project$50 cap is unchanged.


### Verified future-request pricing amendment

On2026-11-04 the official model page https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash lists$0.22/$0.66 per million uncached input/output tokens for the exact requested model ID. Future worker processes use these verified rates instead of the older conservative$0.30/$1.20 assumptions. Historical costs, settled unknown billing and prices recorded in already reserved requests remain unchanged. No cached-input savings are assumed. The$50 cap remains unchanged. Evidence and scope: runtime/DEEPSEEK_PRICING_AMENDMENT.json.


### Manual selected-source review and unnamed-gradient correction

Manual inspection of22 selected sources found an additional residual-feature gradient step in Nemotron diffusion seed0's first replacement (mutation004). That replacement and its costs are retained but excluded; runtime/EXPERIMENT_REPAIRS.json points to a second fresh20-proposal condition. The validator now also traces unnamed residual-feature correlations reaching weights, with a closed-form-solve exception for permitted fitting. It remains conservative and can reject legal code. Source review distinguishes such flags from established violations. All selected sources now require recorded Codex source inspection before any final tests, even with zero heuristic flags. The other21 inspected selected programs were recorded as permitted statistics/constructions. The system prompt now explicitly prohibits single manually calculated gradient/Newton updates; this clarification applies to newly started workers. Completed earlier runs retain their original prompt, so cross-model diffusion comparisons include this method-compliance amendment. Fresh compact classification comparisons all use the same amended prompt. Eight affected conditions and all original seeds/budgets remain in the matrix.


### GPU scheduling amendment

After the diffusion queue completes, its worker on GPU1 helps claim remaining classification jobs. GPU0 continues classification throughout. Exclusive per-job filesystem locks prevent duplicate dispatches, skip active jobs and keep prefix materialization private until complete. Existing children finish before old dispatchers retire. Both devices are matching RTX4500 Ada GPUs; all task/model/seed/proposal budgets, data and selection rules remain unchanged. The shared reservation ledger still enforces the inclusive$50 cap. This changes wall-clock scheduling only.
