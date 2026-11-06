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


def extract_losses(archive):
    result = json.loads((ROOT/'results/results.json').read_text())
    jobs, baselines = {}, {}
    for row in result['classification']+result['diffusion']:
        key = '/'.join(str(row[k]) for k in ['group', 'task', 'model', 'seed'])
        folder = (archive/row['weights']).parent.parent
        metadata = json.loads((folder/'prefixes'/f"budget{row['budget']:03d}.json").read_text())
        metric = 'mse' if row['task'] == 'diffusion' else 'cross_entropy'
        job = jobs.setdefault(key, {k:row[k] for k in ['group', 'task', 'model', 'seed']})
        job.setdefault('points', []).append(dict(budget=row['budget'], loss=metadata['selection'][metric], score=metadata['selection']['score'], weights_sha256=metadata['weights_sha256']))
        baseline_key = f"{row['task']}/{row['seed']}"
        if baseline_key not in baselines:
            path = archive/f"diffusion_runs/baselines/seed{row['seed']}/adam_result.json" if row['task'] == 'diffusion' else archive/f"rich_runs/rich_main/{row['task']}/seed{row['seed']}/adam_result.json"
            adam = json.loads(path.read_text())
            axis = 'step' if row['task'] == 'diffusion' else 'epoch'
            lr = adam['best']['lr']
            baselines[baseline_key] = dict(task=row['task'], seed=row['seed'], lr=lr, axis=axis,
                selected_x=adam['best'][axis], selected_loss=adam['best'][metric],
                points=[dict(x=r[axis], loss=r[metric]) for r in adam['history'] if r['lr'] == lr], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for job in jobs.values():
        row = next(r for r in result['classification']+result['diffusion'] if all(r[k] == job[k] for k in ['group', 'task', 'model', 'seed']))
        path = (archive/row['weights']).parent.parent/'loss_curve.jsonl'
        if path.exists():
            metric = 'mse' if job['task'] == 'diffusion' else 'cross_entropy'
            observed = {p['budget']:dict(p) for p in job['points']}
            for line in path.read_text().splitlines():
                record = json.loads(line)
                observed[record['mutation_calls']] = dict(budget=record['mutation_calls'], loss=record['selection'][metric], training_loss=record['feedback'][metric], candidate=record['candidate'])
            job['observations'] = [observed[n] for n in sorted(observed)]
            job['observations_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = dict(split='selection', n_examples=1000, scope='Frozen GEPA prefixes, per-mutation full-training/development observations during continuations, and Adam development histories. Before continuation only the original declared prefixes were logged. No training, inference or API calls in this export.',
                   selection='Classification checkpoint selection uses the recorded composite score, not minimum cross-entropy. Diffusion selection minimizes MSE. Adam shows each seed’s selected-learning-rate trial, not all hyperparameter trials.',
                   jobs=list(jobs.values()), baselines=baselines)
    exported = ROOT/'results/loss_curves.json'
    exported.write_text(json.dumps(payload, separators=(',', ':'))+'\n')
    manifest_path = ROOT/'results/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files']['results/loss_curves.json'] = dict(sha256=hashlib.sha256(exported.read_bytes()).hexdigest(), original_path=None, note='Recorded development losses, unchanged frozen-prefix selections and Adam histories.')
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    plot_losses(payload)


def plot_losses(data):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(deepseek='#205c4b', minimax='#b65337', nemotron='#5e738d')
    names = dict(deepseek='DeepSeek', minimax='MiniMax', nemotron='Nemotron')
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False, 'axes.labelcolor':'#24332e', 'text.color':'#24332e', 'axes.edgecolor':'#a6ad9f'})
    for task, view in [(t, v) for t in ['mnist', 'fashion', 'sst2'] for v in ['matched', 'budget']]+[('diffusion', 'matched')]:
        group = 'diffusion' if task == 'diffusion' else 'models' if view == 'matched' else 'budget'
        selected = [j for j in data['jobs'] if j['task'] == task and j['group'] == group]
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.1), sharey=True)
        positives = []
        for model in sorted({j['model'] for j in selected}):
            rows = sorted([j for j in selected if j['model'] == model], key=lambda j:j['seed'])
            assert [r['seed'] for r in rows] == [0, 1, 2]
            points = [{p['budget']:p for p in r.get('observations', r['points'])} for r in rows]
            common = sorted(set.intersection(*(set(p) for p in points)))
            x = np.array(common)
            values = np.array([[row[n]['loss'] for n in common] for row in points])
            positives.extend(values.ravel().tolist())
            avg = values.mean(0)
            axes[0].plot(x, avg, 'o-', color=colors[model], label=names[model], lw=1.8, ms=4)
            for y in values:
                axes[0].scatter(x, y, color=colors[model], s=8, alpha=.35)
        for seed, color in enumerate(['#25453e', '#647d6b', '#a1997c']):
            baseline = data['baselines'][f'{task}/{seed}']
            x = [p['x'] for p in baseline['points']]
            y = [p['loss'] for p in baseline['points']]
            positives.extend(y)
            axes[1].plot(x, y, color=color, lw=1.6, label=f"Seed {seed}, lr={baseline['lr']:g}")
            axes[1].scatter([baseline['selected_x']], [baseline['selected_loss']], marker='*', s=90, color=color, edgecolors='white', linewidths=.5, zorder=4)
        assert min(positives) > 0
        lo, hi = min(positives)*.75, max(positives)*1.3
        for ax in axes:
            ax.set_yscale('log'); ax.set_ylim(lo, hi)
            ax.grid(True, which='major', alpha=.22)
            ax.legend(fontsize=8, frameon=False)
        axes[0].set_title('GEPA: validation-selected candidates', loc='left', fontsize=11)
        axes[0].set_xlabel('Mutation requests after one-shot construction')
        axes[0].set_xticks(sorted({p['budget'] for j in selected for p in j['points']}))
        axes[0].set_ylabel('Validation MSE (log scale)' if task == 'diffusion' else 'Validation cross-entropy (log scale)')
        axes[1].set_title('Adam: validation-selected learning rate per seed', loc='left', fontsize=11)
        axes[1].set_xlabel('Optimizer steps within selected-LR trial' if task == 'diffusion' else 'Epoch within selected-LR trial')
        label = 'matched compact prompts' if view == 'matched' else 'original-display DeepSeek budget study'
        fig.suptitle(f'{NAMES[task]} · {label} · same 1,000-example validation pool', fontsize=12, y=.99)
        fig.text(.5, .015, 'GEPA: means, faint dots = individual seeds. Adam: stars mark selected checkpoints. Native x-axes differ; this is not a compute-matched comparison.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .055, 1, .95))
        fig.savefig(BLOG/'assets'/f'loss_{task}_{view}.png', dpi=170, facecolor='white')
        plt.close(fig)


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
            for group, model, budget in [('budget', 'deepseek', 60), ('models', 'deepseek', 60), ('models', 'minimax', 60), ('models', 'nemotron', 60)]:
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
    parser.add_argument('--losses-from', type=Path, help='Optional original archive: export recorded development loss histories and plots')
    args = parser.parse_args()
    if args.predictions_from:
        extract_predictions(args.predictions_from)
    if args.losses_from:
        extract_losses(args.losses_from)
    data = json.loads((ROOT / "results/results.json").read_text())
    BLOG.mkdir(exist_ok=True)
    (BLOG / "data.js").write_text("window.STUDY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    examples = (ROOT/'results/predictions.json').read_text().strip()
    (BLOG/'predictions.js').write_text('window.PREDICTIONS = '+examples+';\n')
    losses = (ROOT/'results/loss_curves.json').read_text().strip()
    (BLOG/'losses.js').write_text('window.LOSSES = '+losses+';\n')
    rows = []
    for row in data["classification_summary"] + data["diffusion_summary"]:
        diffusion = row["group"] == "diffusion"
        result = (f'{row["mse_mean"]:.5f} ± {row["mse_sd"]:.5f}' if diffusion else
                  f'{100*row["accuracy_mean"]:.2f}% ± {100*row["accuracy_sd"]:.2f} pp')
        rows.append('<tr data-group="{group}" data-task="{task}" data-model="{model}"><td>{group}</td><td>{task_name}</td><td>{model}</td><td>{budget}</td><td>{result}</td><td>${cost:.4f}</td></tr>'.format(
            **row, task_name=html.escape(NAMES[row["task"]]), result=result, cost=row["mean_cumulative_api_cost"]))
    template = (ROOT / "scripts/templates/blog.html").read_text()
    losses = json.loads((ROOT/'results/loss_curves.json').read_text())
    coverage = []
    for task in ['mnist', 'fashion', 'sst2', 'diffusion']:
        jobs = [j for j in losses['jobs'] if j['task'] == task and j['group'] != 'budget']
        assert {(j['model'], j['seed']) for j in jobs} == {(m, s) for m in ['deepseek', 'minimax', 'nemotron'] for s in [0, 1, 2]}
        count = {len(j['observations']) for j in jobs}
        assert len(count) == 1
        progress = '0 / 5 / 10, then every mutation 11–60' if task != 'diffusion' else '0 / 5 / 10 / 20, then every mutation 21–60'
        adam = [losses['baselines'][f'{task}/{seed}'] for seed in [0, 1, 2]]
        endpoints = ' / '.join(str(max(p['x'] for p in b['points'])) for b in adam)
        axis = 'epochs' if task != 'diffusion' else 'steps'
        metric = 'cross-entropy' if task != 'diffusion' else 'noise-prediction MSE'
        coverage.append(f'<tr><td>{NAMES[task]}</td><td>{metric}</td><td>{progress}<br>{next(iter(count))} measured points per LLM and seed</td><td>Three seed curves, ending at {endpoints} {axis}</td></tr>')
    comparisons = []
    for task in ['mnist', 'fashion', 'sst2']:
        adam = sum(r['accuracy'] for r in data['baselines'] if r['task'] == task)/3
        for model in ['deepseek', 'minimax', 'nemotron']:
            cells = {r['budget']:r['accuracy_mean'] for r in data['classification_summary'] if r['group'] == 'models' and r['task'] == task and r['model'] == model}
            comparisons.append(f"<tr><td>{NAMES[task]}</td><td>{model}</td><td>{100*cells[10]:.2f}%</td><td>{100*cells[60]:.2f}%</td><td>{100*(cells[60]-cells[10]):+.2f} pp</td><td>{100*adam:.2f}%</td></tr>")
    (BLOG / "index.html").write_text(template.replace("<!-- RESULT_ROWS -->", "\n".join(rows)).replace('<!-- FOLLOWUP_ROWS -->', '\n'.join(comparisons)).replace('<!-- LOSS_COVERAGE_ROWS -->', '\n'.join(coverage)))
    assets = BLOG / "assets"
    assets.mkdir(exist_ok=True)
    for row in data["diffusion"] + [r for r in data["baselines"] if r["task"] == "diffusion"]:
        source = ROOT / "results/generations" / Path(row["generation"]["grid"]).name
        shutil.copy2(source, assets / source.name)
    print(f"Built blog: {len(rows)} aggregate rows, {len(data['classification'])+len(data['diffusion'])} budget measurements, cached examples and diffusion grids; no API calls.")


if __name__ == "__main__":
    main()
