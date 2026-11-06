"""Verify retained evidence and article data without inference or API calls."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text())


def main():
    manifest = read('results/manifest.json')
    for name, record in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == record['sha256'], name
    assert len(manifest['checkpoints']) == 57
    followup = manifest['followup']
    result = read('results/results.json')
    assert len(result['classification']) == followup['classification_rows'] == 207
    assert len(result['diffusion']) == followup['diffusion_rows'] == 54
    assert len(result['baselines']) == 12
    summary = result['classification_summary'] + result['diffusion_summary']
    assert len(summary) == followup['summary_rows'] == 87
    for row in summary:
        source = result['diffusion'] if row['group'] == 'diffusion' else result['classification']
        seeds = [r for r in source if all(r[k] == row[k] for k in ['group', 'task', 'model', 'budget'])]
        assert sorted(r['seed'] for r in seeds) == [0, 1, 2]
        metric = 'mse' if row['group'] == 'diffusion' else 'accuracy'
        values = [r[metric] for r in seeds]
        assert all(math.isfinite(v) for v in values)
        avg = sum(values)/3
        sd = math.sqrt(sum((v-avg)**2 for v in values)/2)
        assert math.isclose(avg, row[metric+'_mean'], abs_tol=1e-12)
        assert math.isclose(sd, row[metric+'_sd'], abs_tol=1e-12)
    reliability = result['reliability']
    assert len(reliability) == 45
    assert sum(r['mutation_budget'] for r in reliability) == followup['mutation_slots'] == 2700
    assert sum(r['valid_mutation_programs'] for r in reliability) == followup['valid_mutations'] == 1436
    assert sum(not r['construction_succeeded'] for r in reliability) == 20
    ledger = read('results/audit/usage.json')
    assert len(ledger['calls']) == followup['requests'] == 3619
    assert ledger['spent'] <= ledger['cap'] == result['accounting']['cap']
    assert result['accounting']['extension_spent'] <= result['accounting']['extension_cap'] == 40
    assert abs(ledger['reserved']) < 1e-9
    assert math.isclose(ledger['spent'], result['accounting']['whole_project_cost'], abs_tol=1e-9)
    page = (ROOT/'blog/index.html').read_text()
    data = (ROOT/'blog/data.js').read_text()
    assert json.loads(data.removeprefix('window.STUDY = ').removesuffix(';\n')) == result
    assert len(re.findall(r'<tr data-group=', page)) == 87
    examples = read('results/predictions.json')
    script = (ROOT/'blog/predictions.js').read_text()
    assert json.loads(script.removeprefix('window.PREDICTIONS = ').removesuffix(';\n')) == examples
    selections = read('results/audit/EXPANSION_SELECTIONS_FROZEN.json')['selections']
    extended = read('results/audit/FINAL_SELECTIONS_AND_REVIEW_FROZEN.json')['selections']
    def selected(group, task, model, seed, budget):
        if group == 'budget':
            return selections[f'deepseek60_{task}_{seed}'][str(budget)]
        return extended[f'{group}_{model}_{task}_{seed}'][str(budget)]
    reviewed = read('results/audit/FOLLOWUP_MECHANISM_REVIEW.json')
    receipt = read('results/audit/FOLLOWUP_COMPLETION.json')
    assert len(receipt['selected_source_sha256']) == followup['selected_sources'] == 165
    assert all(reviewed[s]['verdict'] == 'allowed_statistics' for s in receipt['selected_source_sha256'])
    for checkpoint in manifest['checkpoints']:
        if checkpoint['group'] != 'adam':
            assert checkpoint['budget'] == 60
            assert checkpoint['sha256'] == selected(checkpoint['group'], checkpoint['task'], checkpoint['model'], checkpoint['seed'], 60)['weights_sha256']
    for task in ['mnist', 'fashion', 'sst2']:
        section = examples['tasks'][task]
        assert [r['index'] for r in section['samples']] == list(range(24))
        assert len(section['conditions']) == 12
        for condition in section['conditions']:
            group, model, seed, budget = [condition[k] for k in ['group', 'model', 'seed', 'budget']]
            for method, prefix in [('one_shot', 0), ('gepa', budget)]:
                expected = next(r for r in result['classification'] if all(r[k] == v for k, v in dict(group=group, task=task, model=model, seed=seed, budget=prefix).items()))
                assert condition[method]['accuracy'] == expected['accuracy']
                assert condition[method]['weights_sha256'] == selected(group, task, model, seed, prefix)['weights_sha256']
            expected = next(r for r in result['baselines'] if r['task'] == task and r['seed'] == seed)
            assert condition['adam']['accuracy'] == expected['accuracy']
            checkpoint = next(r for r in manifest['checkpoints'] if r['group'] == 'adam' and r['task'] == task and r['seed'] == seed)
            assert condition['adam']['weights_sha256'] == checkpoint['sha256']
            for method in ['one_shot', 'gepa', 'adam']:
                predictions = condition[method]['predictions']
                assert len(predictions) == 24 and all(0 <= v < len(section['labels']) for v in predictions)
    losses = read('results/loss_curves.json')
    script = (ROOT/'blog/losses.js').read_text()
    assert json.loads(script.removeprefix('window.LOSSES = ').removesuffix(';\n')) == losses
    assert losses['split'] == 'selection' and losses['n_examples'] == 1000
    assert len(losses['jobs']) == 45 and len(losses['baselines']) == 12
    assert sum(len(j['points']) for j in losses['jobs']) == 261
    for task in ['mnist', 'fashion', 'sst2', 'diffusion']:
        jobs = [j for j in losses['jobs'] if j['task'] == task and j['group'] != 'budget']
        assert {(j['model'], j['seed']) for j in jobs} == {(m, s) for m in ['deepseek', 'minimax', 'nemotron'] for s in [0, 1, 2]}
        expected = {0, 5, 10, *range(20, 61)} if task == 'diffusion' else {0, 5, *range(10, 61)}
        for job in jobs:
            observed = {p['budget']:p for p in job['observations']}
            assert set(observed) == expected
            for point in job['points']:
                assert math.isclose(point['loss'], observed[point['budget']]['loss'], rel_tol=1e-5, abs_tol=1e-6)
            for point in observed.values():
                if point['budget'] >= (20 if task == 'diffusion' else 10):
                    assert math.isfinite(point['training_loss']) and point['training_loss'] > 0
        for seed in [0, 1, 2]:
            baseline = losses['baselines'][f'{task}/{seed}']
            assert baseline['points'] and baseline['axis'] == ('step' if task == 'diffusion' else 'epoch')
    for job in losses['jobs']:
        task, model, seed = job['task'], job['model'], job['seed']
        metric = 'mse' if task == 'diffusion' else 'cross_entropy'
        for point in job['points']:
            frozen = selected(job['group'], task, model, seed, point['budget'])
            assert point['loss'] == frozen['selection'][metric]
            assert point['weights_sha256'] == frozen['weights_sha256']
            assert math.isfinite(point['loss']) and point['loss'] > 0
        if job['group'] != 'budget':
            assert job['observations'][-1]['budget'] == 60
            assert all(math.isfinite(p['loss']) and p['loss'] > 0 for p in job['observations'])
    for baseline in losses['baselines'].values():
        chosen = next(p for p in baseline['points'] if p['x'] == baseline['selected_x'])
        assert chosen['loss'] == baseline['selected_loss']
        assert all(math.isfinite(p['loss']) and p['loss'] > 0 for p in baseline['points'])
    for target in re.findall(r'(?:href|src)="([^"#]+)"', page):
        if not target.startswith(('http:', 'https:')):
            assert (ROOT/'blog'/target.split('#')[0]).exists(), target
    print('PASS: artifact hashes, checkpoints, aggregate results, cached prediction examples, accounting and blog.')


if __name__ == '__main__':
    main()
