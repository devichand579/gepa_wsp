# GEPA × Weight-Space Programming

Final code and results: **45 retained trajectories, 3 LLMs, seeds 0/1/2, 4 tasks, 60 mutations, $77.58 total API cost**. The completed extension cost **$31.76 / $40**, including excluded continuations and replacements.

**[Visual blog](blog/index.html)** · **[Final findings](results/FINDINGS.md)** · [Results](results/results.json) · [CSV](results/summary.csv) · [Prediction examples](results/predictions.json) · [Loss curves](results/loss_curves.json)

The loop improves one-shot statistical weight programs, but does not generally replace Adam. Closed-form fits and training feedback are allowed; parameter-gradient updates are excluded. Text results often use bag-of-words constructions; diffusion remains weak. Full findings and limitations are in the blog.

| Path | Purpose |
|---|---|
| `run.py` | Prepare data, train Adam, run GEPA, evaluate frozen weights |
| `src/` | Final models, reflection, sandbox, accounting and continuation runner |
| `results/` | All metrics, sample arrays, 45 final LLM weights, 12 Adam weights and audit receipts |
| `blog/` | Offline interactive article |
| `tests/`, `scripts/` | Checks and article builder |

```bash
# View the blog, or open blog/index.html directly.
python3 -m http.server 8765 --bind 127.0.0.1
# http://127.0.0.1:8765/blog/

# Verify retained evidence and rebuild the article. No model calls.
python3 scripts/verify.py
python3 scripts/build_blog.py
```

Experiments require Linux/bubblewrap and the packages in `requirements.txt`. The recorded environment used Python 3.10 and CUDA 12.8; install compatible Torch/torchvision builds for your device.

```bash
python run.py prepare --task all
python run.py adam --task mnist --seed 0

# New paid run: separate work/ directory and explicit inclusive cap.
export FIREWORKS_API_KEY_FILE=/path/to/private/keyfile
python run.py search --task mnist --model deepseek --seed 0 --mutations 10 --cap 1

python run.py evaluate --task mnist --seed 0 \
  --weights results/checkpoints/budget/mnist/deepseek/seed0.npz

python -m unittest discover -s tests -p 'test_*.py'
```

Use `--device cpu` for CPU execution. `--workspace /path` before the command selects another data/output directory. Search defaults to the matched compact prompt; `--full-context` selects the original classification budget-study display. Final evaluation is a separate command; review new selected programs before interpreting their results as method-compliant.

The 60-mutation follow-up continues all 27 compact classification searches from 10 and all 9 diffusion searches from 20, using seeds 0/1/2 and an additional $40 inclusive API cap. It requires the original full archive, not just endpoint weights:

```bash
python run.py --workspace /path/to/continuations resume --archive /path/to/original-study --cap 40
python run.py --workspace /path/to/continuations status
```

Four background workers share two GPUs and a locked ledger retaining historical spending. Candidate pools and adapter state resume through native GEPA0.1.4; its sampler RNG restarts from the experiment seed. Observational loss logs record full training and development metrics after each mutation. All selections must finish and pass source review before final evaluation. Existing test results are already known, so this is an explicitly declared follow-up study.

November 6 completion: all 36 continuations and three necessary replacements finished at 60 mutations. Source review excluded derivative-guided calibration, a softmax bias-gradient step and Newton calibration, repeating affected continuations from clean original 10-mutation pools before new tests. The amended screen caught all 13 known prohibited programs across 177 reviewed examples without new false positives among reviewed allowed sources. Every selected source was inspected before its corresponding final evaluation. The blog now includes 261 held-out checkpoint rows, all endpoints at 60 mutations, cached prediction examples and measured continuation loss curves.

The earlier study remains in the [original complete snapshot](https://github.com/devichand579/gepa_wsp/tree/5b4c4bf50725d250f38a0e309833abbc46083373). The local continuation archives preserve full programs, native state, excluded trajectories and replacements; the condensed checkout includes endpoint weights, metrics, review decisions and source hashes in [the manifest](results/manifest.json). Refactored source is a final implementation, not the byte-identical historical source freeze. Exact continuation reproduction requires the full local archives and the recorded repair mapping, beyond endpoint weights alone. Credentials and raw datasets are excluded; the blog includes the fixed first 24 held-out examples per classification task with cached one-shot/GEPA/Adam predictions.
