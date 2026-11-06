# GEPA × WSP: final 60-mutation findings

Completed November 6, 2026. Three LLMs, four tasks, seeds 0/1/2. The matched classification trajectories continued from 10 to 60 mutations; diffusion continued from 20 to 60. The separate original-display DeepSeek study remains in the full results.

The additional budget produced modest classification gains and larger diffusion improvements. Adam still won on both image classifiers and denoising. DeepSeek and MiniMax exceeded this Adam baseline on SST-2, often by encoding bag-of-words statistical classifiers in the transformer container. This does not establish learning of attention.

| Task | LLM | 10 mutations | 60 mutations ± seed SD | Change (pp) | Adam |
|---|---|---:|---:|---:|---:|
| MNIST | deepseek | 83.27% | 86.89% ± 3.36 pp | +3.63 | 92.31% |
| MNIST | minimax | 84.77% | 85.07% ± 3.60 pp | +0.30 | 92.31% |
| MNIST | nemotron | 82.41% | 82.41% ± 0.91 pp | -0.00 | 92.31% |
| Fashion-MNIST | deepseek | 80.92% | 81.10% ± 2.51 pp | +0.18 | 84.14% |
| Fashion-MNIST | minimax | 82.36% | 82.73% ± 1.54 pp | +0.36 | 84.14% |
| Fashion-MNIST | nemotron | 77.04% | 77.04% ± 1.64 pp | +0.00 | 84.14% |
| SST-2 | deepseek | 73.32% | 74.08% ± 1.11 pp | +0.76 | 70.57% |
| SST-2 | minimax | 72.90% | 73.66% ± 0.93 pp | +0.76 | 70.57% |
| SST-2 | nemotron | 53.13% | 53.02% ± 1.70 pp | -0.11 | 70.57% |

Image scores use the official 10,000-example test splits. SST-2 uses its 872-example official labeled validation split; the official test labels are unavailable. Every new numeric checkpoint was selected and reviewed before this follow-up evaluation, but the original final-split results were already known when the follow-up was proposed. This is an adaptive exploratory study, not fresh confirmatory evidence.

| Denoising LLM | 20-mutation MSE | 60-mutation MSE ± seed SD | Relative reduction |
|---|---:|---:|---:|
| deepseek | 0.27113 | 0.25518 ± 0.04567 | 5.9% |
| minimax | 0.84117 | 0.50622 ± 0.20608 | 39.8% |
| nemotron | 1.00299 | 0.53842 ± 0.40145 | 46.3% |

Adam denoising MSE: **0.09161**. The best LLM endpoint is still 2.79× this error. Generation uses 128 fixed draws per seed, showing the first 64 without curation. No validated perceptual metric was computed; improved MSE alone does not establish useful generation.

## Validation protocol and loss curves

The displayed curves use the fixed 1,000-example validation pool, called `selection` or `development` in the saved files. They are not final-test loss curves. Every task includes DeepSeek, MiniMax and Nemotron through 60 mutations, each with seeds 0/1/2, plus three Adam seed histories. Classification has 53 measured points per LLM/seed (0, 5, 10, then every mutation 11–60); diffusion has 44 (0, 5, 10, 20, then every mutation 21–60). Missing earlier intermediate losses are not reconstructed. GEPA curves track the selected candidate rather than every proposal; rejected proposals can leave the loss flat. Adam curves display each seed's complete winning-learning-rate history, with a star at its selected checkpoint. Their native progress axes are different and do not match computation.

| Task | Training feedback | Validation used for selection | Final scoring split |
|---|---:|---:|---|
| MNIST | 2,000 official-training images | 1,000 other official-training images | 10,000 official test images |
| Fashion-MNIST | 2,000 official-training images | 1,000 other official-training images | 10,000 official test images |
| SST-2 | 4,000 official-training reviews | 1,000 other official-training reviews | 872 official labeled validation reviews |
| MNIST diffusion | 2,000 clean training images | 1,000 other clean training images | 10,000 official MNIST test images with fresh noise |

Training and validation IDs are disjoint within each seed. Image training/validation pools have 200/100 examples per class; SST-2 has 2,000/500 per class. These fixed pools are shared by all LLM conditions and Adam within the seed. Validation examples and traces do not enter reflection or generated programs; validation scores influence GEPA's candidate and parent selection. Repeated use can overfit this pool. Seeds reuse the same final examples, not independent final datasets.

Adam classification tests learning rates 1e−4, 1e−3 and 1e−2, up to 50 epochs per trial, batch size 128 and patience 10. It selects by the same composite validation score as GEPA, with lower cross-entropy breaking ties; the selected checkpoint need not minimize cross-entropy. Adam diffusion runs 1,000 steps at each of 1e−3 and 3e−4, batch size 64, validates every 100 steps and selects the lowest validation MSE. All tuning trials consume compute even though the figures show only the winning learning-rate histories.

The SST-2 vocabulary is fitted only on feedback reviews (2,000 tokens including PAD/UNK, maximum length 64). Held-out review inputs were inspected for duplicate checks during preparation; normalized and encoded cross-split overlap counts were zero. Final labels and predictions did not guide fitting or checkpoint selection. Diffusion reuses the MNIST split IDs and fixed validation noise at seed 82,000 + experiment seed; final noise uses 83,000 + seed. Adam refreshes training noise, whereas LLM programs use a fixed feedback noise bank.

Every declared checkpoint and selected source was frozen and reviewed before its corresponding final evaluation. Earlier final results nevertheless informed the decision to extend to 60 mutations. Reusing that final split makes the follow-up exploratory rather than confirmatory; all declared prefixes are reported instead of selecting a prefix by final score.

## What the method allows

Generated NumPy programs receive full permitted training arrays, weights and forward diagnostics. Training-only statistics, direct ridge/LDA/least-squares solves, PCA, augmentation, internal forward passes and finite candidate calibration are allowed. Autograd, manual parameter-loss gradients, numerical parameter gradients and Newton/Hessian parameter updates are excluded. Consequently these results measure reflection and selection around data-fitting programs, not coefficient choice without numerical fitting.

GEPA selection uses a disjoint development pool. Selection inputs/labels/traces do not enter reflection or generated code. Classification selects by 0.9 × correctness + 0.1 × true-label probability; this can reduce cross-entropy while accuracy stays flat or falls. Native resume preserves candidates and evaluations but restarts sampler RNG from the seed, so it is not identical to an uninterrupted search.

## Review and reliability

Source review excluded three follow-up continuations: DeepSeek Fashion-MNIST seed 2 (derivative-guided temperature bisection), Nemotron Fashion-MNIST seed 1 (softmax bias-gradient step), and MiniMax SST-2 seed 2 (Newton calibration). Each was repeated from its clean original 10-mutation pool, under the same model settings and shared cap, before new final-test evaluation. Original invalid artifacts and their costs remain in the private archive and ledger.

All 115 distinct selected programs in the follow-up were inspected; combined with the retained original-display trajectories, the reported study uses 165 approved selected sources. A static screen alone cannot prove absence of arbitrary optimizer implementations. The amended screen was checked against 177 reviewed examples, catching all 13 known prohibited sources without new false positives among reviewed allowed sources.

Across the 45 retained trajectories, **1,436/2,700 mutation programs** passed execution validation. **20/45 constructors failed**, including all nine MiniMax constructors; their gains include recovery from failed submissions. Valid execution is not equivalent to GEPA acceptance, selection or improvement.

## Cost and interpretation

**Extension: $31.7552 / $40**, across 1,860 additional requests, including the three excluded 50-call continuations and their three replacements. **Whole project: $77.5762**, 3,619 requests, including pilots, compatibility failures, discarded trajectories and repairs. Unknown billing is charged at its reserved upper bound. Costs exclude GPU/CPU and agent-authoring costs. Nested cumulative prefix costs overlap and must not be summed.

Equal request budgets do not match internal compute: a generated program can perform many training-only fits in its 60-CPU-second, 4-GiB sandbox and is executed twice to check determinism. Adam refreshed training noise; LLM diffusion programs used a fixed training noise bank. There was no reasoning-effort ablation; every LLM requested thinking enabled with an 8,192-token budget. Differences in reasoning-token use do not establish that more reasoning caused better performance.

Three seeds and reused public datasets support a small exploratory finding. Greedy reflection, random proposals and hand-written statistical-fitting controls are still needed to identify the contribution of GEPA selection. Parameter counts describe fixed containers, not proven learning in every layer.

Full frozen rows: [results.json](results.json). [Aggregate CSV](summary.csv). [Visual blog](../blog/index.html). [Selection and review gate](audit/FINAL_SELECTIONS_AND_REVIEW_FROZEN.json). [Completion receipt](audit/FOLLOWUP_COMPLETION.json).
