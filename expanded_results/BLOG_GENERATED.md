# Reflective weight programming across budgets, LLMs and diffusion

The LLM writes feedback-aware NumPy weight-update programs for fixed neural
networks. Full weights and permitted training arrays are available to each program;
statistics, covariance/PCA and closed-form fitting are allowed. GEPA supplies paired
feedback evaluation and Pareto candidate selection. This measures the whole
reflection-and-selection strategy around a generated statistical program. It does
not isolate GEPA's Pareto mechanism or show that coefficients were chosen without
numerical algorithms. See the generated-program audit before claiming optimizer
absence.

## Predeclared comparisons

The original-representation DeepSeek study uses60 mutation calls, with
prefixes0,5,10,30,60. Eligible original trajectories continue from30 calls.
Fresh matched compact-prompt runs compare DeepSeek,
MiniMax M3.1 Flash and Nemotron Lightning3.5 at0,5,10 calls. All use seeds0,1,2.
MNIST uses a101770-parameter MLP; Fashion-MNIST a20490-parameter CNN; SST-2 an
83266-parameter transformer. The diffusion target is a56609-parameter convolutional
epsilon predictor with a fixed32-step cosine DDPM on MNIST16x16. Its three LLMs
are compared at0,5,10,20 calls, with Adam trained separately.

Before final-test evaluation, source review found prohibited explicit parameter
gradient/Newton updates in original DeepSeek MNIST seeds0/1/2, Fashion-MNIST
seeds1/2 and Nemotron diffusion seeds0/1/2. Those full trajectories were excluded and replaced by fresh
constructions and searches with the same models, data, seeds and full budgets.
Original artifacts and costs are retained. The replacements use a pre-execution
method screen, followed by selected-lineage review; a static screen alone cannot
prove that every possible optimizer implementation is absent. The replacement
mapping is recorded in runtime/EXPERIMENT_REPAIRS.json. No test results informed
this repair. The affected classification budget prefixes belong to fresh60-call runs.

Parameter counts describe the fixed network containers, not how many parameters
were usefully fitted. Source inspection found SST-2 programs that zero attention
and feedforward blocks and encode a bag-of-words ridge classifier in embeddings
and the head. Other programs fit only a readout over fixed or constructed
features. Such results demonstrate compilation of statistical procedures into
network weights; they do not by themselves establish gradient-free learning of
attention or every hidden layer. The proposal budgets count LLM requests, including
failed submissions. Each generated program may perform many training-only fits,
augmentations and forward passes within a60-CPU-second/4-GiB sandbox limit; the
program is executed twice to check determinism. Equal request counts do not imply
equal FLOPs, data exposure or parameter-update counts. The Adam comparison does
not match this internal computation. These statistical fits still constitute
learning from data; the optimizer restriction excludes parameter-gradient methods,
not all numerical fitting algorithms.

Both one-shot and GEPA have the same initial access to all permitted training
arrays and initial forward outputs. GEPA starts from that construction and gets
additional reflection/evaluation calls. Classification selects by0.9*correct+
0.1*p_true; diffusion by negative MSE. Selection data influences search but never
enters reflection or generated code. Every expanded selection was frozen before
final-test access. Budget prefixes exclude later candidates and follow the SAME
trajectory; they are not independent runs. More budget need not improve test
accuracy, and the composite selection objective need not yield monotone accuracy.

## Final held-out results

| Study | Task | Model | Calls | Mean final result ± seed SD | Mean cumulative API cost |
|---|---|---|---:|---:|---:|
| budget | fashion | deepseek | 0 | 64.38% ± 33.86 pp | $0.0439 |
| budget | fashion | deepseek | 5 | 79.06% ± 8.43 pp | $0.2686 |
| budget | fashion | deepseek | 10 | 79.11% ± 8.25 pp | $0.4946 |
| budget | fashion | deepseek | 30 | 77.66% ± 5.67 pp | $1.3984 |
| budget | fashion | deepseek | 60 | 81.09% ± 4.05 pp | $2.7592 |
| budget | mnist | deepseek | 0 | 84.88% ± 6.25 pp | $0.0196 |
| budget | mnist | deepseek | 5 | 85.09% ± 6.60 pp | $0.1229 |
| budget | mnist | deepseek | 10 | 86.26% ± 6.64 pp | $0.2274 |
| budget | mnist | deepseek | 30 | 86.75% ± 4.58 pp | $0.6460 |
| budget | mnist | deepseek | 60 | 87.43% ± 5.01 pp | $1.2718 |
| budget | sst2 | deepseek | 0 | 62.73% ± 7.86 pp | $0.0545 |
| budget | sst2 | deepseek | 5 | 66.90% ± 2.35 pp | $0.3178 |
| budget | sst2 | deepseek | 10 | 70.34% ± 2.16 pp | $0.6097 |
| budget | sst2 | deepseek | 30 | 73.51% ± 0.94 pp | $1.6780 |
| budget | sst2 | deepseek | 60 | 72.52% ± 0.95 pp | $2.9393 |
| models | fashion | deepseek | 0 | 78.88% ± 6.65 pp | $0.0153 |
| models | fashion | deepseek | 5 | 78.88% ± 6.65 pp | $0.0953 |
| models | fashion | deepseek | 10 | 80.92% ± 2.26 pp | $0.1783 |
| models | fashion | minimax | 0 | 18.22% ± 7.78 pp | $0.0134 |
| models | fashion | minimax | 5 | 61.52% ± 36.44 pp | $0.0821 |
| models | fashion | minimax | 10 | 82.36% ± 2.00 pp | $0.1517 |
| models | fashion | nemotron | 0 | 18.22% ± 7.78 pp | $0.0037 |
| models | fashion | nemotron | 5 | 54.03% ± 38.24 pp | $0.0221 |
| models | fashion | nemotron | 10 | 77.04% ± 1.64 pp | $0.0402 |
| models | mnist | deepseek | 0 | 80.30% ± 2.65 pp | $0.0163 |
| models | mnist | deepseek | 5 | 83.27% ± 4.39 pp | $0.1022 |
| models | mnist | deepseek | 10 | 83.27% ± 4.39 pp | $0.1880 |
| models | mnist | minimax | 0 | 11.26% ± 1.35 pp | $0.0137 |
| models | mnist | minimax | 5 | 61.47% ± 42.14 pp | $0.0826 |
| models | mnist | minimax | 10 | 84.77% ± 2.05 pp | $0.1545 |
| models | mnist | nemotron | 0 | 35.10% ± 39.98 pp | $0.0044 |
| models | mnist | nemotron | 5 | 82.17% ± 0.99 pp | $0.0248 |
| models | mnist | nemotron | 10 | 82.41% ± 0.91 pp | $0.0452 |
| models | sst2 | deepseek | 0 | 52.06% ± 1.40 pp | $0.0139 |
| models | sst2 | deepseek | 5 | 73.32% ± 1.07 pp | $0.1337 |
| models | sst2 | deepseek | 10 | 73.32% ± 1.07 pp | $0.2994 |
| models | sst2 | minimax | 0 | 50.38% ± 1.82 pp | $0.0098 |
| models | sst2 | minimax | 5 | 72.90% ± 1.29 pp | $0.0691 |
| models | sst2 | minimax | 10 | 72.90% ± 1.29 pp | $0.1889 |
| models | sst2 | nemotron | 0 | 52.48% ± 2.93 pp | $0.0020 |
| models | sst2 | nemotron | 5 | 53.13% ± 1.85 pp | $0.0114 |
| models | sst2 | nemotron | 10 | 53.13% ± 1.85 pp | $0.0207 |
| diffusion | MNIST16 | deepseek | 0 | MSE 6.07430 ± 10.04958 | $0.0243 |
| diffusion | MNIST16 | deepseek | 5 | MSE 0.28216 ± 0.03105 | $0.1430 |
| diffusion | MNIST16 | deepseek | 10 | MSE 0.28216 ± 0.03105 | $0.2826 |
| diffusion | MNIST16 | deepseek | 20 | MSE 0.27113 ± 0.05005 | $0.5487 |
| diffusion | MNIST16 | minimax | 0 | MSE 1.00319 ± 0.00276 | $0.0126 |
| diffusion | MNIST16 | minimax | 5 | MSE 1.00319 ± 0.00276 | $0.0857 |
| diffusion | MNIST16 | minimax | 10 | MSE 0.84117 ± 0.28243 | $0.1662 |
| diffusion | MNIST16 | minimax | 20 | MSE 0.84117 ± 0.28243 | $0.3232 |
| diffusion | MNIST16 | nemotron | 0 | MSE 1.00315 ± 0.00279 | $0.0037 |
| diffusion | MNIST16 | nemotron | 5 | MSE 1.00299 ± 0.00296 | $0.0258 |
| diffusion | MNIST16 | nemotron | 10 | MSE 1.00299 ± 0.00296 | $0.0517 |
| diffusion | MNIST16 | nemotron | 20 | MSE 1.00299 ± 0.00296 | $0.1038 |

![Budget study](figures/expanded/budget.png)

![Matched model study](figures/expanded/models.png)

![Diffusion denoising](figures/expanded/diffusion.png)

Three seeds are exploratory. Dots/error bars summarize seed variation, not all
LLM sampling uncertainty. Conditional paired test-example intervals and unadjusted
McNemar p-values are in expanded_results/results.json. The same final split is
reused across seeds; these are not independent test datasets. Image final splits
have10000 examples; SST-2 uses its872-example official labeled validation split.
A greedy-reflection control would be needed to attribute improvements specifically
to Pareto selection.

## Diffusion samples

Final noise draws and generation seeds differ from reflection feedback. Adam
refreshes training noise each optimizer step; the LLM conditions use their fixed
paired training noise bank. All share the same clean-image pool. This is not an
equal-compute or identical-noise-exposure comparison. Lower denoising MSE alone
is not evidence of useful generation. All fixed-seed sample arrays/grids are saved;
these pixel summaries are not FID or validated perceptual metrics.


nemotron, seed0, budget0:

![Fixed first64 draws](expanded_results/generations/seed0_843bd57eceaed0e132299178def7ec05e3218e96edf5913dc65c1afca2712b99.png)

nemotron, seed0, budget20:

![Fixed first64 draws](expanded_results/generations/seed0_0d9cd6db775b77f783fc66c59882b990c65ebd6c66dd63d826c49213be5561eb.png)

minimax, seed0, budget0:

![Fixed first64 draws](expanded_results/generations/seed0_843bd57eceaed0e132299178def7ec05e3218e96edf5913dc65c1afca2712b99.png)

minimax, seed0, budget20:

![Fixed first64 draws](expanded_results/generations/seed0_7ea2debefcad6df9465cc299774869bcd2bcc944beaa6e77e70b78a780ff768d.png)

deepseek, seed0, budget0:

![Fixed first64 draws](expanded_results/generations/seed0_2fe44c55d02461ade3752c58c30ba75c763255f7c2f864a9504a0843dff6b49d.png)

deepseek, seed0, budget20:

![Fixed first64 draws](expanded_results/generations/seed0_df2f41f901b8293205cfd7d50b0bfa1ee6a562505290295e2a09cbb505e63148.png)

Adam, seed0:

![Adam fixed first64 draws](expanded_results/generations/seed0_9fd01073dfb25844892974193db89b8c47371c638ad00250c586f9eb126af163.png)

## Reliability, mechanism and cost

Construction failures retain random initialization; their gains include recovery
from a failed submission. Provider errors and invalid programs consume call slots.
Context-rejected compatibility runs are retained separately and included in whole-
project accounting. Compact display prompts solve the smaller model's context
limit; full-precision arrays inside edit(w,f) are unchanged. Original DeepSeek
budget trajectories retain their original representation, while the separate
fresh model comparison uses the same compact representation for all three LLMs.

The audit checks feedback-example provenance, selection trace boundaries, source
and checkpoint hashes, and credential artifacts, and lists generated-program calls
and loops for mechanism review. Absence of a named optimizer API is not proof that
no manual gradient loop was written. Retained random features, zeroed attention
branches and nonzero parameter counts also need careful interpretation: nominal
architecture alone does not establish learned attention or all-layer learning.
Public-dataset familiarity in the LLM pretraining cannot be ruled out.

Cumulative cost rows reuse earlier calls; do not sum costs across nested budgets.
Whole-project accounting includes every historical pilot and discarded condition.
Unknown billing is charged at its full reserved upper bound. No new paid inference
is performed in this evaluator. See EXPANSION_PLAN.md, runtime/EXPANSION_FROZEN.json,
expanded_results/audit.json and the preserved requests/programs/checkpoints.

Motivation: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [GEPA](https://arxiv.org/abs/2507.19457). These experiments test a different
combination rather than reproduce their headline results.

### Proposal reliability

These counts pool the full three-seed trajectories once, rather than counting nested budget prefixes repeatedly. A valid program need not be accepted or improve a score.

| Study | Task | LLM | Valid mutation programs / slots | Failed one-shot constructions | Mutation failure breakdown |
|---|---|---|---:|---:|---|
| budget | fashion | deepseek | 145/180 | 1 | code_or_array_error: 20, method_rejection: 1, other_failure: 14 |
| budget | mnist | deepseek | 159/180 | 0 | code_or_array_error: 14, method_rejection: 2, other_failure: 5 |
| budget | sst2 | deepseek | 127/180 | 0 | code_or_array_error: 12, other_failure: 21, provider_failure: 1, resource_limit: 1, response_truncation: 18 |
| diffusion | diffusion | deepseek | 36/60 | 0 | code_or_array_error: 10, other_failure: 5, resource_limit: 5, response_truncation: 4 |
| diffusion | diffusion | minimax | 3/60 | 3 | code_or_array_error: 42, other_failure: 8, resource_limit: 2, response_truncation: 5 |
| diffusion | diffusion | nemotron | 3/60 | 2 | code_or_array_error: 27, method_rejection: 8, other_failure: 8, resource_limit: 3, response_truncation: 11 |
| models | fashion | deepseek | 23/30 | 0 | code_or_array_error: 7 |
| models | fashion | minimax | 5/30 | 3 | code_or_array_error: 21, method_rejection: 1, other_failure: 3 |
| models | fashion | nemotron | 14/30 | 3 | code_or_array_error: 16 |
| models | mnist | deepseek | 26/30 | 0 | code_or_array_error: 4 |
| models | mnist | minimax | 6/30 | 3 | code_or_array_error: 23, other_failure: 1 |
| models | mnist | nemotron | 27/30 | 1 | code_or_array_error: 1, other_failure: 2 |
| models | sst2 | deepseek | 23/30 | 1 | other_failure: 6, response_truncation: 1 |
| models | sst2 | minimax | 6/30 | 3 | code_or_array_error: 14, method_rejection: 1, other_failure: 7, resource_limit: 2 |
| models | sst2 | nemotron | 19/30 | 0 | code_or_array_error: 10, other_failure: 1 |

Conservatively accounted project API cost: $45.8210 / $50; requests: 1759. Retained program audit flags, including rejected proposals: 10. Every selected source passed recorded inspection before final tests.
