"""Text provenance audit and held-out evaluation after every search is frozen."""
import json,math,re
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
import text_experiment as t
import experiment as common
ROOT=common.ROOT

def audit():
 key=Path('/tmp/gepa-weight-space-fireworks.key').read_text().strip();count=0
 for path in (ROOT/'text_runs').rglob('request.json'):
  raw=path.read_text();assert key not in raw
  request=json.loads(raw);assert [m['role'] for m in request['messages']]==['system','user']
  user=request['messages'][1]['content'];assert 'selection_index' not in user and 'test_index' not in user
  assert '"split": "selection"' not in user and '"split": "test"' not in user
  assert all(0<=int(i)<4000 for i in re.findall(r'"feedback_index"\s*:\s*(\d+)',user))
  seed=int(path.parts[path.parts.index('sst2')+1].removeprefix('seed'));t.install(seed)
  fb=common.load_npz(ROOT/'data/sst2'/f'seed{seed}'/'feedback.npz')
  start=user.index('{"network":');payload=json.JSONDecoder().raw_decode(user[start:])[0]
  def check(value):
   if isinstance(value,dict):
    if 'feedback_index' in value:
     i=value['feedback_index'];assert value['label']==int(fb['y'][i])
     if 'tokenized_text' in value:assert value['tokenized_text']==t.decode(fb['x'][i])
    for v in value.values():check(v)
   elif isinstance(value,list):
    for v in value:check(v)
  check(payload);count+=1
 for path in (ROOT/'text_runs').rglob('*'):
  if path.is_file() and path.suffix in ['.json','.jsonl','.md','.log']:assert key.encode() not in path.read_bytes()
 common.dump(ROOT/'runtime/text_audit.json',{'passed':True,'requests_checked':count,'feedback_example_content_verified':True})

def finalize():
 roots=[ROOT/'text_runs/sst2'/f'seed{s}' for s in [0,1,2]]
 for root in roots:
  for name in ['search_result.json','adam_result.json','one_shot.npz','gepa_best.npz','adam_best.npz']:assert (root/name).exists(),f'Text test locked: {root/name}'
 audit()
 common.dump(ROOT/'text_runs/SELECTION_FROZEN.json',{str(r.relative_to(ROOT)):json.loads((r/'search_result.json').read_text())['best_candidate'] for r in roots})
 reports=[]
 for seed,root in enumerate(roots):
  t.install(seed);data=common.load_npz(ROOT/'data/final_test/sst2'/f'seed{seed}.npz')
  states={m:common.load_npz(root/f) for m,f in [('one_shot','one_shot.npz'),('gepa','gepa_best.npz'),('adam','adam_best.npz')]}
  results={m:t.evaluate_state('sst2',s,data['x'],data['y'],'cuda:0') for m,s in states.items()}
  a=results['one_shot']['pred']==data['y'];b=results['gepa']['pred']==data['y'];d=b.astype(float)-a
  gained=int((~a&b).sum());lost=int((a&~b).sum());se=float(d.std(ddof=1)/math.sqrt(len(d)))
  initial=t.initial('sst2',seed)
  groups={m:{'total_parameters':sum(v.size for k,v in states['gepa'].items() if k.startswith(m+'.')),
              'nonzero_parameters':sum(int(np.count_nonzero(v)) for k,v in states['gepa'].items() if k.startswith(m+'.')),
              'changed_from_initial':sum(int(np.count_nonzero(v!=initial[k])) for k,v in states['gepa'].items() if k.startswith(m+'.')),
              'changed_from_construction':sum(int(np.count_nonzero(v!=states['one_shot'][k])) for k,v in states['gepa'].items() if k.startswith(m+'.'))} for m in t.modules('sst2')}
  report={'task':'sst2','seed':seed,'n_test':len(d),'accuracy':{m:r['accuracy'] for m,r in results.items()},'cross_entropy':{m:r['cross_entropy'] for m,r in results.items()},'paired_gain':float(d.mean()),'gain_normal_95ci':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],'wrong_to_right':gained,'right_to_wrong':lost,'mcnemar_exact_p':float(binomtest(gained,gained+lost,p=.5).pvalue) if gained+lost else 1.,'parameter_groups':groups}
  common.dump(root/'final_test.json',report);np.savez(root/'final_predictions.npz',y=data['y'],**{m:r['pred'] for m,r in results.items()});reports.append(report)
 aggregate={m:{'mean':float(np.mean([r['accuracy'][m] for r in reports])),'sd':float(np.std([r['accuracy'][m] for r in reports],ddof=1))} for m in ['one_shot','gepa','adam']}
 common.dump(ROOT/'text_runs/FINAL_RESULTS.json',{'runs':reports,'aggregate':aggregate,'uncertainty_note':'Three exploratory seeds on the same final split. Paired intervals condition on each selected model.'})
 table=['| Seed | One-shot | GEPA | Adam | Gain |','|---|---:|---:|---:|---:|']
 for r in reports:
  a=r['accuracy'];table.append(f'| {r["seed"]} | {a["one_shot"]:.1%} | {a["gepa"]:.1%} | {a["adam"]:.1%} | {r["paired_gain"]*100:+.2f} pp |')
 text='# Strict weight edits on a sentiment transformer\n\nAn 83,266-parameter transformer with two width-32 blocks, two attention heads and vocabulary2000. No pretrained target weights or tokenizer. See TEXT_PROTOCOL.md for the frozen method.\n\n'+ '\n'.join(table)+'\n\nFinal held-out split: official SST-2 validation, 872 examples. All three searches were frozen before inference. One-shot received16 feedback texts and GEPA received additional feedback; do not attribute differences solely to Pareto selection. Parameter-group counts in text_runs/FINAL_RESULTS.json show which parts were changed or zeroed. Accuracy alone does not establish learned attention. All malformed submissions and partial edits are retained.\n'
 (ROOT/'BLOG_TRANSFORMER.md').write_text(text);print(json.dumps(aggregate))
if __name__=='__main__':torch.set_num_threads(2);finalize()
