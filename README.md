# GEPA × Weight-Space Programming

A completed exploratory study of LLM-written statistical weight programs inside GEPA's reflection-and-selection loop: **45 trajectories, three LLMs, seeds 0/1/2, four target networks, 162 held-out budget checkpoints and 12 Adam baselines**. Inclusive project API spending: **$45.82 of $50**, including failed and discarded runs.

**[Read the visual blog](blog/index.html)** · [Detailed report](BLOG_EXPANDED.md) · [Results JSON](expanded_results/results.json) · [Summary CSV](expanded_results/summary.csv)

![Research blog preview](blog/assets/article-preview.png)

The repository is private. The blog is included as local static files; no public site or Pages deployment is configured. To view it, clone the repository, then either open `blog/index.html` directly in a browser or serve the checkout:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
# Open http://127.0.0.1:8765/blog/
```

The blog works offline, with interactive budget/task controls, all three LLM comparisons, diffusion grids for each seed, reliability and cost breakdowns, and a filterable table. It uses no external fonts, scripts, analytics or model APIs.

## Findings

DeepSeek's original-representation budget study, held-out means over three seeds:

| Target | One shot | GEPA / 30 mutations | GEPA / 60 mutations | Adam |
|---|---:|---:|---:|---:|
| MNIST MLP | 84.88% | 86.75% | 87.43% | 92.31% |
| Fashion-MNIST CNN | 64.38% | 77.66% | 81.09% | 84.14% |
| SST-2 transformer container | 62.73% | 73.51% | 72.52% | 70.57% |

Reflection improved one-shot programs but did not generally replace Adam. Many text programs disabled attention and encoded bag-of-words classifiers. Diffusion final MSE at 20 mutations was **0.27113 / 0.84117 / 1.00299** for DeepSeek / MiniMax / Nemotron, versus **0.09161** for Adam; LLM samples remained poor. All nine MiniMax one-shot constructions in the separate matched classification comparison failed validation, so its later gains include recovery from broken submissions. Across the expanded study, **622/990 mutation programs** passed execution validation and **20/45 constructors failed**.

Three seeds are exploratory. Prefixes share trajectories. No greedy-reflection or hand-written statistical-fitting control isolates GEPA's contribution, and internal computation/data exposure are not matched to Adam.

## What the method allows

The LLM writes NumPy `edit(w, f)` programs with current weights, permitted training arrays and forward diagnostics. Statistics, PCA, covariance, closed-form fitting and internal forward passes are permitted. **Parameter-gradient and Newton updates are excluded from approved selected programs.** These programs still learn from data using numerical algorithms.

Programs run in a network-free sandbox with 60 CPU seconds and 4 GiB, twice for determinism. GEPA uses paired feedback batches and Pareto selection, edits all modules jointly and disables crossover. Disjoint selection examples never enter reflection or edit programs. Final tests opened only after selections were frozen and all 115 distinct selected-lineage programs passed recorded source inspection. Eight original conditions required fresh replacements after method violations; one replacement required another repair. Costs remain included.

## Repository map

| Path | Contents |
|---|---|
| [`blog/`](blog/) | Visual research article; browser-ready HTML/CSS/JS and local assets |
| [`expanded_results/`](expanded_results/) | All held-out metrics, 54-row summary, audit and fixed diffusion sample arrays |
| [`runtime/`](runtime/) | Frozen selections, review receipts, repair mapping and inclusive usage ledger |
| [`ARTIFACT_MANIFEST.json`](ARTIFACT_MANIFEST.json) | SHA-256 hashes of 1,695 byte-identical exported research files |
| `expansion_runs/`, `diffusion_runs/`, `rich_runs/` | Frozen checkpoints, generated program sources and selected run metadata |
| [`RICH_PROTOCOL.md`](RICH_PROTOCOL.md), [`EXPANSION_PLAN.md`](EXPANSION_PLAN.md) | Feedback-aware method, comparisons and amendments |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Safe local checks, environments, snapshot scope and replay limits |

Original experimental source files remain byte-identical to the final source freeze. New article and snapshot verification code lives in `scripts/`. Earlier pilot plans/results and strict-protocol documents are historical; they are not pooled with the completed expanded study. The old full working directory is preserved separately on the research machine.

## Verify or rebuild without inference

Python's standard library is sufficient:

```bash
python3 scripts/verify_snapshot.py
python3 scripts/build_blog.py
```

The snapshot intentionally excludes raw datasets, raw request/response/prompt payloads, credentials, virtual environments and most unselected checkpoint history. It is an evidence snapshot with research code, not a complete execution-state backup. The full archive remains local. Do not use the archived schedulers as a one-command reproduction recipe; see [reproduction details](REPRODUCIBILITY.md).

Motivation: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and [GEPA](https://arxiv.org/abs/2507.19457). This study tests their proposed combination; it does not reproduce their headline experiments.
