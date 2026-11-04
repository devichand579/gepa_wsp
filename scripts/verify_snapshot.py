"""Read-only integrity and consistency checks for the curated evidence snapshot."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    manifest = read('ARTIFACT_MANIFEST.json')['files']
    for name, record in manifest.items():
        assert sha(name) == record['sha256'], f'Exported research file changed: {name}'
    result = read('expanded_results/results.json')
    assert len(result['classification']) == 126 and len(result['diffusion']) == 36
    assert len(result['baselines']) == 12
    summaries = result['classification_summary'] + result['diffusion_summary']
    assert len(summaries) == 54
    for row in summaries:
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
    for row in result['classification'] + result['diffusion']:
        metadata = read(str(Path(row['weights']).with_suffix('.json')))
        assert sha(row['weights']) == metadata['weights_sha256']
        assert row['n_test'] == (872 if row['task'] == 'sst2' else 10000)
    audit = read('expanded_results/audit.json')
    selected = {r['source_sha256']: r for r in audit['programs'] if r['in_selected_lineage']}
    assert len(selected) == 115
    review = read('runtime/EXPANSION_MECHANISM_REVIEW.json')
    for digest, record in selected.items():
        assert sha(record['source']) == digest
        assert review[digest]['verdict'] == 'allowed_statistics'
    freeze = read('runtime/EXPANSION_FROZEN.json')
    for name, digest in {**freeze['source_hashes'], **freeze['data_hashes']}.items():
        assert sha(name) == digest
    reliability = result['reliability']
    assert len(reliability) == 45
    assert sum(r['mutation_budget'] for r in reliability) == 990
    assert sum(r['valid_mutation_programs'] for r in reliability) == 622
    assert sum(not r['construction_succeeded'] for r in reliability) == 20
    ledger = read('runtime/usage.json')
    assert len(ledger['calls']) == 1759 and ledger['spent'] <= ledger['cap'] == 50
    assert abs(ledger['reserved']) < 1e-9
    assert math.isclose(ledger['spent'], result['accounting']['whole_project_cost'], abs_tol=1e-9)
    for row in result['diffusion'] + [r for r in result['baselines'] if r['task'] == 'diffusion']:
        for field in ['samples', 'grid']:
            assert (ROOT / row['generation'][field]).is_file()
    js = (ROOT / 'blog/data.js').read_text()
    assert json.loads(js.removeprefix('window.STUDY = ').removesuffix(';\n')) == result
    page = (ROOT / 'blog/index.html').read_text()
    assert len(re.findall(r'<tr data-group=', page)) == 54
    for target in re.findall(r'(?:href|src)="([^"#]+)"', page):
        if target.startswith(('http:', 'https:')):
            continue
        assert (ROOT / 'blog' / target.split('#')[0]).exists(), f'Broken article link: {target}'
    print(f'PASS: {len(manifest)} historical hashes; 162 checkpoints; 54 aggregates; 115 approved selected sources; reliability, accounting, blog data and links.')


if __name__ == '__main__':
    main()
