# Reproduction and evidence scope

## Read or rebuild the article

Open `blog/index.html` in a browser. Data is loaded from a local JavaScript file, so opening through `file://` works without a server. Alternatively run `python3 -m http.server 8765 --bind 127.0.0.1` from the repository and visit `http://127.0.0.1:8765/blog/`. The server is local to the machine.

```bash
python3 scripts/verify_snapshot.py
python3 scripts/build_blog.py
```

Neither command reads held-out examples or makes API calls. The builder copies frozen result values into `blog/data.js`, renders the 54 aggregate rows and copies fixed sample grids. Editing prose happens in `scripts/templates/blog.html`; styles and interactions live in `blog/style.css` and `blog/app.js`.

## Evidence retained

`ARTIFACT_MANIFEST.json` records byte hashes of exported historical files. It is not a hash of newly authored publication files. The exported evidence includes all 960 program-source records in the audit (some paths repeat), all 162 result checkpoint files and their prefix-selection metadata, 12 Adam checkpoint files, every stored diffusion sample array/grid, held-out metrics and the project usage ledger. Generated code includes rejected proposals; only sources marked `in_selected_lineage` and covered by the pretest review receipt support the approved method claim.

`runtime/COMPLETION_AUDIT.json` is the original local completion receipt. `verify_completed_study.py` is the original full-archive verifier: it expects private archive state, timestamps and omitted files and is **not** the snapshot verifier. Use `scripts/verify_snapshot.py` in a clone.

Raw MNIST/Fashion-MNIST/SST-2 datasets, raw request/response/prompt payloads, virtual environments, temporary logs, GEPA serialized state and most unselected checkpoint history are excluded. These remain in the original local study directory. The clone cannot reproduce every provenance check or resume the searches without restoring that full state. No license grant for third-party datasets is implied by their preparation scripts.

## Historical execution environment

The study used Python 3.10.19, two CUDA GPUs, Linux, bubblewrap (`bwrap`), Torch 2.10.0+cu128, torchvision 0.25.0+cu128, NumPy 2.2.6, SciPy 1.15.3, GEPA 0.1.4 and Matplotlib 3.10.9. The original pins are preserved in `requirements.txt`. SST-2 preparation additionally imports `pyarrow` and `huggingface_hub`; Pillow is needed by the plotting/image stack. The historical environment inventory is recorded in `environment.txt` (distribution metadata omits the CUDA build suffix). The original pins are CUDA-specific and are not a universal CPU installation recipe.

On a compatible machine, install the matching Torch/torchvision CUDA builds using the official PyTorch package index, install the remaining packages, and install bubblewrap from the operating system package manager. A CPU-compatible Torch build can run many harness checks but will not reproduce CUDA timing or necessarily bit-identical predictions. No environment installation or scheduler is launched by the blog or snapshot checks.

Useful harness checks, once those dependencies are installed:

```bash
python check_parameter_gradient_screen.py
python check_diffusion_gepa.py
python check_expansion_queue.py
python check_expansion_finalizer.py
python check_expansion_evaluation_flow.py
python check_expansion_report.py
```

These do not make paid API calls. The parameter-gradient check uses nine preserved rejected-program examples; they are not approved evidence. The other checks above use synthetic fixtures. Some use the sandbox and therefore require Linux/bubblewrap. `python check_rich.py` additionally needs the three prepared training feedback pools and the SST-2 vocabulary; restore or regenerate those before running it. The optional browser check `scripts/check_blog_browser.py` additionally requires Playwright and an installed Chromium, and writes screenshots outside the repository unless a destination is supplied.

## Re-evaluating frozen weights

Data preparation is in `prepare.py`, `prepare_text.py` and `diffusion_core.py`. Seed-specific split/vocabulary manifests are retained under `data/`. Image preparation downloads MNIST and Fashion-MNIST; text preparation downloads `stanfordnlp/sst2`. The study used seeds 0, 1, 2; the manifest IDs, rather than a fresh random split, define the intended pools. Upstream unpinned dataset downloads can change, so compare manifests and hashes with the archive before claiming an exact replay.

Frozen LLM checkpoints are at each result row's `weights` path in `expanded_results/results.json`. Classification Adam weights are `rich_runs/rich_main/<task>/seed<seed>/adam_best.npz`; diffusion Adam weights are `diffusion_runs/baselines/seed<seed>/adam_best.npz`. Network definitions and forward passes are in `experiment.py`, `text_experiment.py` and `diffusion_core.py`. The archived `expansion_finalize.py` documents evaluation, but also expects complete frozen search state. Do not run it against this partial snapshot expecting a fresh replay.

The official image test splits have 10,000 examples. SST-2 uses its **872-example labeled validation split** as the held-out endpoint; the official test split is unlabeled. Diffusion final noise uses seed 83,000 + experiment seed; generation uses 89,000 + seed. All final sample arrays contain 128 draws; displayed grids are the first 64.

## Starting new paid experiments

The preserved runner sources read a credential from `/tmp/gepa-weight-space-fireworks.key` outside the repository. The file's path is not the credential; its contents are excluded. The completed `runtime/usage.json` contains the inclusive historical spending ledger and must not be reset, reused as an empty budget, or silently repriced. New experiments need a separate working directory, fresh explicit cap, new ledger and an updated protocol. The archived continuation scheduler expects old 30-call histories and is not a general clean-start launcher.

Use `rich_experiment.py` / `rich_worker.py` for feedback-aware classification, `text_experiment.py` for the SST-2 target, `diffusion_experiment.py` for the regression adapter and `expansion_client.py` for provider settings. `repair_experiment.py` documents the fresh-replacement route. Method screen decisions do not replace full selected-source review. New experiments must keep final examples outside reflection and program execution, freeze selections, inspect selected programs, then evaluate.

## Interpretation limits

One-shot construction sees the same permitted initial feedback as later proposals. Budget counts mutation requests, including failures, after that first request. Prefixes are correlated, not independent runs. Internal fitting, augmentation and repeated forward passes are not counted as LLM requests or matched against Adam. Adam refreshes diffusion training noise; the LLM program uses a fixed paired training bank. The three-seed results and per-example conditional comparisons are exploratory. See the visual article for the full reliability, repair and mechanism caveats.
