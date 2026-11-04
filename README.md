# GEPA × Weight-Space Programming

Final code and results: **45 runs, 3 LLMs, seeds 0/1/2, 4 network tasks, $45.82 API cost**.

**[Visual blog](blog/index.html)** · [Results](results/results.json) · [CSV](results/summary.csv) · [Prediction examples](results/predictions.json)

The loop improves one-shot statistical weight programs, but does not generally replace Adam. Closed-form fits and training feedback are allowed; parameter-gradient updates are excluded. Text results often use bag-of-words constructions; diffusion remains weak. Full findings and limitations are in the blog.

| Path | Purpose |
|---|---|
| `run.py` | Prepare data, train Adam, run GEPA, evaluate frozen weights |
| `src/` | Nine final modules: models, reflection, sandbox, fitting interface and accounting |
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

Reported metrics are unchanged. Historical receipts and result paths refer to the [original complete snapshot](https://github.com/devichand579/gepa_wsp/tree/5b4c4bf50725d250f38a0e309833abbc46083373), which preserves intermediate weights, generated programs and the full recorded protocol. The current checkout contains only endpoint weights; their paths and hashes are in [the manifest](results/manifest.json). Refactored source is a final implementation, not the byte-identical historical source freeze. Credentials and full raw datasets are excluded; the blog includes the first 24 held-out examples per classification task, with cached one-shot/GEPA/Adam predictions.
