"""Read-only completion audit. Does not fit models, call APIs, or rerun inference."""
import csv
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
from PIL import Image
import expansion_finalize as final

ROOT=Path(__file__).resolve().parent
result=json.loads((ROOT/'expanded_results/results.json').read_text())
jobs=final.validate_gate()
audit=json.loads((ROOT/'expanded_results/audit.json').read_text())
final.require_mechanism_review(audit)
assert len(jobs)==45
assert [len(result[k]) for k in ['classification','diffusion','baselines','reliability']]==[126,36,12,45]
for kind in ['classification','diffusion']:
 rows=result[kind]
 for key in {(r['group'],r['task'],r['model'],r['budget']) for r in rows}:
  assert sorted(r['seed'] for r in rows if (r['group'],r['task'],r['model'],r['budget'])==key)==[0,1,2],key
 for row in rows:
  assert np.isfinite(row['accuracy'] if kind=='classification' else row['mse'])
  assert row['n_test']==(872 if row['task']=='sst2' else 10000)
  if kind=='classification':assert row['parameter_count']=={'mnist':101770,'fashion':20490,'sst2':83266}[row['task']]
with (ROOT/'expanded_results/summary.csv').open() as handle:assert len(list(csv.DictReader(handle)))==54
for row in result['reliability']:
 assert row['valid_mutation_programs']+row['failed_mutation_proposals']+row['unclassified_slots']==row['mutation_budget']
usage=json.loads((ROOT/'runtime/usage.json').read_text())
assert not usage['blocked'] and abs(usage['reserved'])<1e-9
assert usage['spent']<=50 and len(usage['calls'])<=2000
assert not any(c['status'] in {'pending','unknown_billing'} for c in usage['calls'])
assert abs(sum(c.get('cost',0) for c in usage['calls'])-usage['spent'])<1e-7
assert abs(result['accounting']['whole_project_cost']-usage['spent'])<1e-9
receipt=json.loads((ROOT/'runtime/SELECTED_SOURCE_INSPECTION_COMPLETE.json').read_text())
methods=json.loads((ROOT/'runtime/EXPANSION_METHODS_REVIEWED.json').read_text())
ready=json.loads((ROOT/'runtime/EXPANDED_RESULTS_READY.json').read_text())
assert receipt['selected_sources']==receipt['selected_sources_with_recorded_allowed_decisions']==115
assert receipt['time']<methods['time']<ready['time']
cache=list((ROOT/'expanded_results/cache').glob('*'))
assert cache and min(p.stat().st_mtime for p in cache)>=receipt['time']
review_path=ROOT/'runtime/EXPANSION_MECHANISM_REVIEW.json'
assert hashlib.sha256(review_path.read_bytes()).hexdigest()==receipt['review_manifest_sha256']
reviews=json.loads(review_path.read_text())
for program in audit['programs']:
 if program['in_selected_lineage']:
  assert reviews[program['source_sha256']]['verdict']=='allowed_statistics'
  assert hashlib.sha256((ROOT/program['source']).read_bytes()).hexdigest()==program['source_sha256']
configuration=json.loads((ROOT/'runtime/EXPANSION_FROZEN.json').read_text())
for name,sha in configuration.get('data_hashes',{}).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
for document in ['README.md','CURRENT_EXPERIMENT.md','BLOG_EXPANDED.md']:
 for target in re.findall(r'\]\(([^)]+)\)',(ROOT/document).read_text()):
  if not target.startswith(('https://','http://','#')):assert (ROOT/target.split('#')[0]).exists(),(document,target)
for path in (ROOT/'figures/expanded').glob('*.png'):
 with Image.open(path) as img:img.verify()
grids=set()
for row in result['diffusion']+[r for r in result['baselines'] if r['task']=='diffusion']:
 generation=row['generation'];grids.add(generation['grid'])
 with Image.open(ROOT/generation['grid']) as img:img.verify()
 with np.load(ROOT/generation['samples'],allow_pickle=False) as arrays:
  samples=arrays[arrays.files[0]]
  assert samples.shape==(128,1,16,16) and np.isfinite(samples).all()
active=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if (proc/'cwd').resolve()!=ROOT:continue
  args=[a.decode() for a in (proc/'cmdline').read_bytes().split(b'\0') if a]
 except (OSError,UnicodeDecodeError):continue
 if any(Path(a).name in {'repair_experiment.py','classification_followup.py','diffusion_experiment.py','expansion_queue.py'} for a in args):active.append(int(proc.name))
assert not active,active
record={'time':time.time(),'passed':True,'jobs':45,'classification_rows':126,'diffusion_rows':36,'baseline_rows':12,'summary_rows':54,
 'selected_sources_reviewed_before_tests':115,'whole_project_cost':usage['spent'],'requests':len(usage['calls']),'pending_requests':0,
 'unique_generation_grids_checked':len(grids),'final_test_cache_created_after_source_review':True,
 'frozen_sources_data_manifests_and_checkpoints_verified':True,'blog_images_and_local_links_verified':True,'experiment_workers_active':0,
 'scope':'All declared budgets, models, seeds and tasks; held-out metrics, source gate, accounting, reports, sample arrays and plots.'}
(ROOT/'runtime/COMPLETION_AUDIT.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
