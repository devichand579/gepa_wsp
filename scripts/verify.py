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
    result = read('results/results.json')
    assert len(result['classification']) == 126 and len(result['diffusion']) == 36
    assert len(result['baselines']) == 12
    summary = result['classification_summary'] + result['diffusion_summary']
    assert len(summary) == 54
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
    assert sum(r['mutation_budget'] for r in reliability) == 990
    assert sum(r['valid_mutation_programs'] for r in reliability) == 622
    assert sum(not r['construction_succeeded'] for r in reliability) == 20
    ledger = read('results/audit/usage.json')
    assert len(ledger['calls']) == 1759 and ledger['spent'] <= ledger['cap'] == 50
    assert abs(ledger['reserved']) < 1e-9
    assert math.isclose(ledger['spent'], result['accounting']['whole_project_cost'], abs_tol=1e-9)
    page = (ROOT/'blog/index.html').read_text()
    data = (ROOT/'blog/data.js').read_text()
    assert json.loads(data.removeprefix('window.STUDY = ').removesuffix(';\n')) == result
    assert len(re.findall(r'<tr data-group=', page)) == 54
    examples = read('results/predictions.json')
    script = (ROOT/'blog/predictions.js').read_text()
    assert json.loads(script.removeprefix('window.PREDICTIONS = ').removesuffix(';\n')) == examples
    selections = read('results/audit/EXPANSION_SELECTIONS_FROZEN.json')['selections']
    for task in ['mnist', 'fashion', 'sst2']:
        section = examples['tasks'][task]
        assert [r['index'] for r in section['samples']] == list(range(24))
        assert len(section['conditions']) == 12
        for condition in section['conditions']:
            group, model, seed, budget = [condition[k] for k in ['group', 'model', 'seed', 'budget']]
            job = f'deepseek60_{task}_{seed}' if group == 'budget' else f'{model}_{task}_{seed}'
            for method, prefix in [('one_shot', 0), ('gepa', budget)]:
                expected = next(r for r in result['classification'] if all(r[k] == v for k, v in dict(group=group, task=task, model=model, seed=seed, budget=prefix).items()))
                assert condition[method]['accuracy'] == expected['accuracy']
                assert condition[method]['weights_sha256'] == selections[job][str(prefix)]['weights_sha256']
            expected = next(r for r in result['baselines'] if r['task'] == task and r['seed'] == seed)
            assert condition['adam']['accuracy'] == expected['accuracy']
            checkpoint = next(r for r in manifest['checkpoints'] if r['group'] == 'adam' and r['task'] == task and r['seed'] == seed)
            assert condition['adam']['weights_sha256'] == checkpoint['sha256']
            for method in ['one_shot', 'gepa', 'adam']:
                predictions = condition[method]['predictions']
                assert len(predictions) == 24 and all(0 <= v < len(section['labels']) for v in predictions)
    for target in re.findall(r'(?:href|src)="([^"#]+)"', page):
        if not target.startswith(('http:', 'https:')):
            assert (ROOT/'blog'/target.split('#')[0]).exists(), target
    print('PASS: artifact hashes, checkpoints, aggregate results, cached prediction examples, accounting and blog.')


if __name__ == '__main__':
    main()
