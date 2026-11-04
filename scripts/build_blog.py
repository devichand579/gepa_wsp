"""Build the offline visual article from frozen results; no inference or API calls."""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import io
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / "blog"
NAMES = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "sst2": "SST-2", "diffusion": "MNIST diffusion"}


def extract_predictions(archive):
    """Export the first 24 examples using existing prediction caches, never inference."""
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import hf_hub_download

    def load(path):
        with np.load(path, allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    def digest(arrays):
        sha = hashlib.sha256()
        for k, v in sorted(arrays.items()):
            sha.update(k.encode()); sha.update(str(v.shape).encode())
            sha.update(str(v.dtype).encode()); sha.update(v.tobytes())
        return sha.hexdigest()

    def cached(weights, data, task, seed, accuracy):
        key = hashlib.sha256(json.dumps(dict(weights=digest(load(archive/weights)), test_arrays=digest(data)), sort_keys=True).encode()).hexdigest()
        path = archive/f'expanded_results/cache/class_{task}_{seed}_{key}.npz'
        pred = load(path)['pred']
        assert len(pred) == len(data['y'])
        assert abs(float((pred == data['y']).mean())-accuracy) < 1e-12
        return dict(predictions=pred[:24].tolist(), accuracy=accuracy, cache=path.name, cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), weights_sha256=hashlib.sha256((archive/weights).read_bytes()).hexdigest())

    result = json.loads((ROOT/'results/results.json').read_text())
    raw_path = hf_hub_download('stanfordnlp/sst2', 'data/validation-00000-of-00001.parquet', repo_type='dataset', local_files_only=True)
    text = pq.read_table(raw_path).to_pydict()
    labels = dict(mnist=[str(i) for i in range(10)], fashion=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], sst2=['negative', 'positive'])
    payload = dict(selection='First 24 examples in original held-out split order, fixed across conditions and seeds. Chosen after the study for illustration by index, independently of predictions. Filters operate only within these 24 examples.', source='Existing frozen final-evaluation prediction caches; no model inference, training, selection or API requests.', tasks={})
    for task in ['mnist', 'fashion', 'sst2']:
        section = dict(labels=labels[task], samples=[], conditions=[])
        for seed in [0, 1, 2]:
            data = load(archive/'data/final_test'/(f'sst2/seed{seed}.npz' if task == 'sst2' else task+'.npz'))
            if seed == 0:
                for i in range(24):
                    sample = dict(index=i, label=int(data['y'][i]))
                    if task == 'sst2':
                        assert int(text['idx'][i]) == int(data['ids'][i]) and int(text['label'][i]) == sample['label']
                        sample['text'] = text['sentence'][i]
                    else:
                        buffer = io.BytesIO()
                        Image.fromarray(np.rint(data['x'][i, 0]*255).astype('uint8')).save(buffer, format='PNG')
                        sample['image'] = 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
                    section['samples'].append(sample)
            assert [s['label'] for s in section['samples']] == data['y'][:24].tolist()
            baseline = next(r for r in result['baselines'] if r['task'] == task and r['seed'] == seed)
            adam = cached(Path(f'rich_runs/rich_main/{task}/seed{seed}/adam_best.npz'), data, task, seed, baseline['accuracy'])
            for group, model, budget in [('budget', 'deepseek', 60), ('models', 'deepseek', 10), ('models', 'minimax', 10), ('models', 'nemotron', 10)]:
                rows = [r for r in result['classification'] if r['task'] == task and r['seed'] == seed and r['group'] == group and r['model'] == model]
                first = next(r for r in rows if r['budget'] == 0)
                final = next(r for r in rows if r['budget'] == budget)
                section['conditions'].append(dict(group=group, model=model, budget=budget, seed=seed, construction_failed=bool(first['construction_error']), one_shot=cached(Path(first['weights']), data, task, seed, first['accuracy']), gepa=cached(Path(final['weights']), data, task, seed, final['accuracy']), adam=adam))
        payload['tasks'][task] = section
    exported = ROOT/'results/predictions.json'
    exported.write_text(json.dumps(payload, separators=(',', ':'))+'\n')
    manifest_path = ROOT/'results/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['results/predictions.json'] = dict(sha256=hashlib.sha256(exported.read_bytes()).hexdigest(), original_path=None, note='Post-study illustration from existing frozen prediction caches; fixed first 24 examples per task.')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print('Exported 24 fixed held-out examples per classification task, four conditions and three seeds; cached predictions only.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions-from', type=Path, help='Optional original archive: export cached classification examples without inference')
    args = parser.parse_args()
    if args.predictions_from:
        extract_predictions(args.predictions_from)
    data = json.loads((ROOT / "results/results.json").read_text())
    BLOG.mkdir(exist_ok=True)
    (BLOG / "data.js").write_text("window.STUDY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    examples = (ROOT/'results/predictions.json').read_text().strip()
    (BLOG/'predictions.js').write_text('window.PREDICTIONS = '+examples+';\n')
    rows = []
    for row in data["classification_summary"] + data["diffusion_summary"]:
        diffusion = row["group"] == "diffusion"
        result = (f'{row["mse_mean"]:.5f} ± {row["mse_sd"]:.5f}' if diffusion else
                  f'{100*row["accuracy_mean"]:.2f}% ± {100*row["accuracy_sd"]:.2f} pp')
        rows.append('<tr data-group="{group}" data-task="{task}" data-model="{model}"><td>{group}</td><td>{task_name}</td><td>{model}</td><td>{budget}</td><td>{result}</td><td>${cost:.4f}</td></tr>'.format(
            **row, task_name=html.escape(NAMES[row["task"]]), result=result, cost=row["mean_cumulative_api_cost"]))
    template = (ROOT / "scripts/templates/blog.html").read_text()
    (BLOG / "index.html").write_text(template.replace("<!-- RESULT_ROWS -->", "\n".join(rows)))
    assets = BLOG / "assets"
    assets.mkdir(exist_ok=True)
    for row in data["diffusion"] + [r for r in data["baselines"] if r["task"] == "diffusion"]:
        source = ROOT / "results/generations" / Path(row["generation"]["grid"]).name
        shutil.copy2(source, assets / source.name)
    print("Built blog: 54 aggregate rows, 162 budget measurements, cached examples and diffusion grids; no API calls.")


if __name__ == "__main__":
    main()
