# Weight-space programming through programs with GEPA

GEPA writes each weight program from a reflection on the current candidate. It scores a 128-example feedback batch, then the model reads those metrics, the current weights, layer traces, and recent proposals and returns one NumPy `edit(w, f)`. That program runs once and saves new weights. Closed-form fits and statistics are allowed. Parameter-gradient updates are rejected. Later selection scores the frozen tensors only; the program does not run again. The first program is a one-shot construction from the initial forward pass, before this reflection loop.

`run.py` is the entry point. `src/` is the implementation.

| File | Role |
|---|---|
| `run.py` | Prepare data, train Adam, run GEPA, evaluate frozen weights, and resume a preserved search |
| `src/core.py` | MNIST and Fashion-MNIST networks, metrics, API budget ledger, response parsing, and the Adam baseline |
| `src/classification.py` | Reflection loop for image and text classification: builds the feedback reflection, asks for `edit(w, f)`, and keeps candidates by Pareto selection. Also runs the one-shot construction |
| `src/text_model.py` | Fixed SST-2 transformer and its Adam baseline; installs that network into the shared backend |
| `src/llm.py` | Fireworks client for DeepSeek, MiniMax, and Nemotron: spend reservation, exact-request replay, prompt compaction, and a provider circuit breaker |
| `src/diffusion.py` | Same reflection loop for unconditional diffusion; the score is negative mean squared error on epsilon prediction |
| `src/diffusion_model.py` | 16×16 convolutional DDPM, 32-step cosine schedule, sampling, data preparation from the MNIST splits, and an Adam baseline |
| `src/sandbox.py` | Isolated worker for `edit(w, f)`: NumPy and math only, CPU and memory limits, and a screen against gradient or residual-feature updates |
| `src/prepare.py` | Fixed seeds 0/1/2: 2000 feedback and 1000 selection examples for MNIST, Fashion-MNIST, and SST-2, with a feedback-only 2000-token vocabulary |
| `src/prefixes.py` | Copies the best selection-set checkpoint at each requested mutation budget, using only evaluations finished before the next dispatch |
| `src/resume.py` | Continues preserved searches to 60 proposals under a shared extra spend cap, then freezes prefixes |

Tasks are `mnist`, `fashion`, `sst2`, and `diffusion`. Seeds are `0`, `1`, and `2`. Models are `deepseek`, `minimax`, and `nemotron`.

Experiments need Linux with bubblewrap and the packages in `requirements.txt`. Outputs go to `work/` unless `--workspace` is set.

```bash
python run.py prepare --task all
python run.py adam --task mnist --seed 0

export FIREWORKS_API_KEY_FILE=/path/to/private/keyfile
python run.py search --task mnist --model deepseek --seed 0 --mutations 10 --cap 1

python run.py evaluate --task mnist --seed 0 --weights /path/to/weights.npz

python run.py --workspace /path/to/continuations resume --archive /path/to/original-study --cap 40
python run.py --workspace /path/to/continuations status
```

`--device cpu` runs on CPU. Search uses the compact prompt unless `--full-context` is set; diffusion always uses the compact prompt. `evaluate` scores a saved weight file on the held-out split. `resume` needs the original search archive, including GEPA state, and continues it under an extra API dollar cap.
