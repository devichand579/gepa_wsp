"""Audit rich-feedback provenance, freeze all selections, then evaluate final tests."""
import ast,json,math,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
import rich_experiment as rich
import experiment as common
ROOT=common.ROOT

def audit():
 key=Path('/tmp/gepa-weight-space-fireworks.key').read_text().strip();checked=0;programs=[]
 for task in ['mnist','fashion','sst2']:
  for seed in [0,1,2]:
   b=rich.backend(task,seed);folder=ROOT/'rich_runs/rich_main'/task/f'seed{seed}'
   fb=common.load_npz(ROOT/'data'/task/f'seed{seed}/feedback.npz')
   for path in (folder/'llm').rglob('request.json'):
    raw=path.read_text();assert key not in raw
    request=json.loads(raw);assert [m['role'] for m in request['messages']]==['system','user']
    user=request['messages'][1]['content'];payload=json.JSONDecoder().raw_decode(user[user.index('{"network":'):])[0]
    assert 'selection_index' not in user and 'test_index' not in user
    assert '"split": "selection"' not in user and '"split": "test"' not in user
    for ex in payload['feedback']['examples']:
     i=ex['feedback_index'];assert 0<=i<len(fb['y']) and ex['label']==int(fb['y'][i])
     if 'image_28x28_ascii' in ex:assert ex['image_28x28_ascii']==common.ascii_image(fb['x'][i])
    manifest=json.loads((path.parent/'feedback_manifest.json').read_text())
    assert manifest['split']=='feedback' and manifest['n_full_feedback']==len(fb['y'])
    assert all(0<=i<len(fb['y']) for i in manifest['batch_indices'])
    assert manifest['array_schema']['x']['shape']==list(fb['x'].shape)
    checked+=1
   ev=folder/'evaluations.jsonl'
   assert all(not row['capture_traces'] for row in (json.loads(l) for l in ev.read_text().splitlines()) if row['split']=='selection')
   for path in [folder/'construction.py',*(folder/'llm').rglob('edit.py')]:
    if not path.exists():continue
    source=path.read_text();tree=ast.parse(source)
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    loops=[ast.unparse(n.iter) for n in ast.walk(tree) if isinstance(n,ast.For)]
    flags=[c for c in calls if any(t in c.lower() for t in ['backward','adam','optim','gradient','sgd'])]
    programs.append({'path':str(path.relative_to(ROOT)),'calls':sorted(set(calls)),'for_iterators':loops,'while_loops':sum(isinstance(n,ast.While) for n in ast.walk(tree)),'optimizer_name_flags':flags})
 for path in (ROOT/'rich_runs/rich_main').rglob('*'):
  if path.is_file() and path.suffix in ['.json','.jsonl','.py','.md','.log']:assert key.encode() not in path.read_bytes()
 result={'passed':True,'requests_checked':checked,'feedback_examples_verified':True,'selection_traces_absent':True,'programs':programs,'mechanism_review':'AST summaries aid review; no named optimizer is not a mathematical proof that a manual optimization loop is absent.'}
 common.dump(ROOT/'runtime/RICH_AUDIT.json',result)
 return result

def finalize():
 # Expanded conditions were predeclared before tests: a local wait uses no Codex inference.
 while (ROOT/'runtime/EXPANSION_PLAN.json').exists() and not all((ROOT/'runtime'/name).exists() for name in ['EXPANSION_SELECTIONS_FROZEN.json','EXPANSION_METHODS_REVIEWED.json']):time.sleep(60)
 roots=[ROOT/'rich_runs/rich_main'/task/f'seed{seed}' for task in ['mnist','fashion','sst2'] for seed in [0,1,2]]
 for root in roots:
  for name in ['search_result.json','adam_result.json','one_shot.npz','gepa_best.npz','adam_best.npz']:assert (root/name).exists(),f'Rich final test locked: missing {root/name}'
  assert json.loads((root/'search_result.json').read_text())['proposals']==30
 audit()
 common.dump(ROOT/'rich_runs/rich_main/SELECTION_FROZEN.json',{str(r.relative_to(ROOT)):json.loads((r/'search_result.json').read_text())['best_candidate'] for r in roots})
 reports=[]
 for root in roots:
  task=root.parent.name;seed=int(root.name.removeprefix('seed'));b=rich.backend(task,seed)
  data=common.load_npz(ROOT/'data/final_test'/('sst2/seed'+str(seed)+'.npz' if task=='sst2' else task+'.npz'))
  states={m:common.load_npz(root/name) for m,name in [('one_shot','one_shot.npz'),('gepa','gepa_best.npz'),('adam','adam_best.npz')]}
  results={m:rich.forward(b,task,state,data['x'],data['y'],'cuda:0')[0] for m,state in states.items()}
  a=results['one_shot']['pred']==data['y'];c=results['gepa']['pred']==data['y'];d=c.astype(float)-a
  gained=int((~a&c).sum());lost=int((a&~c).sum());se=float(d.std(ddof=1)/math.sqrt(len(d)))
  initial=b.initial(task,seed);counts={m:{'parameters':sum(v.size for v in state.values()),'nonzero':sum(int(np.count_nonzero(v)) for v in state.values()),'changed_from_initial':sum(int(np.count_nonzero(v!=initial[k])) for k,v in state.items())} for m,state in states.items()}
  report={'task':task,'seed':seed,'n_test':len(d),'accuracy':{m:r['accuracy'] for m,r in results.items()},'cross_entropy':{m:r['cross_entropy'] for m,r in results.items()},'paired_gain':float(d.mean()),'gain_normal_95ci':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],'wrong_to_right':gained,'right_to_wrong':lost,'mcnemar_exact_p':float(binomtest(gained,gained+lost,p=.5).pvalue) if gained+lost else 1.,'parameter_counts':counts,'construction_status':json.loads((root/'construction_status.json').read_text())}
  if task=='sst2':report['zero_output_branches']={block+'.'+branch:all(np.count_nonzero(states['gepa'][block+'.'+branch+'.'+p])==0 for p in ['weight','bias']) for block in ['block0','block1'] for branch in ['proj','ff2']}
  common.dump(root/'final_test.json',report);np.savez(root/'final_predictions.npz',y=data['y'],**{m:r['pred'] for m,r in results.items()});reports.append(report)
 aggregate={task:{m:{'mean':float(np.mean([r['accuracy'][m] for r in reports if r['task']==task])),'sd':float(np.std([r['accuracy'][m] for r in reports if r['task']==task],ddof=1))} for m in ['one_shot','gepa','adam']} for task in ['mnist','fashion','sst2']}
 common.dump(ROOT/'rich_runs/rich_main/FINAL_RESULTS.json',{'runs':reports,'aggregate':aggregate,'note':'Three exploratory seeds; paired intervals condition on selected models. Same final split repeats across seeds, not independent test datasets.'})
 make_blog(reports,aggregate);print(json.dumps(aggregate),flush=True)

def make_blog(reports,aggregate):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 methods=['one_shot','gepa','adam'];labels=['One-shot + feedback','LLM + GEPA','Adam']
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
 for ax,task in zip(axes,['mnist','fashion','sst2']):
  rows=[r for r in reports if r['task']==task];values=np.array([[r['accuracy'][m]*100 for m in methods] for r in rows])
  ax.bar(range(3),values.mean(0),color=['#8398ae','#2685a6','#e4a14a'])
  for i,row in enumerate(values):ax.scatter(np.arange(3)+(i-1)*.05,row,c='black',s=16)
  ax.set_title(task.upper());ax.set_xticks(range(3),labels,rotation=20);ax.set_ylim(0,100)
 axes[0].set_ylabel('Final held-out accuracy (%)');fig.tight_layout();out=ROOT/'figures/rich';out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'accuracy.png',dpi=180);plt.close(fig)
 table=['| Task | Seed | One-shot + feedback | GEPA | Adam | Gain |','|---|---:|---:|---:|---:|---:|']
 for r in reports:
  a=r['accuracy'];table.append(f'| {r["task"]} | {r["seed"]} | {a["one_shot"]:.1%} | {a["gepa"]:.1%} | {a["adam"]:.1%} | {r["paired_gain"]*100:+.2f} pp |')
 ledger=json.loads((ROOT/'runtime/usage.json').read_text());unknown=[i for i,c in enumerate(ledger['calls']) if c['status']=='conservative_settlement']
 blog='''# Can reflection improve feedback-aware neural weight programs?

We gave an LLM the current weights, permitted training data and cached forward
activations, and asked it to write NumPy programs that update a fixed neural
network. GEPA supplied reflection and Pareto candidate selection. Statistics,
covariance calculations and closed-form fitting were allowed; this is programmatic
gradient-free learning, not a claim that every coefficient was chosen without a
numerical algorithm. Review generated programs before claiming optimizer absence.

## Fixed tasks and comparisons

MNIST: a101770-parameter MLP. Fashion-MNIST: a20490-parameter CNN. SST-2: an
83266-parameter transformer with two width32 blocks, two heads and vocabulary2000.
No pretrained target weights or tokenizers. Seeds0,1,2 for every task. Image
training/feedback pools contain2000 examples; text4000. Selection pools contain1000.
Both one-shot construction and GEPA have the same initial access to all feedback
arrays and their initial forward outputs. GEPA starts from that exact construction
and gets30 mutation API calls, including invalid submissions and provider failures.
Adam uses the same initial weights and feedback data, with three learning rates,
up to50 epochs per rate and selection-based checkpoint choice. Extra training and
feedback computation is reported; no equal-compute claim is made.

All parameter modules may be changed in one proposal. Prediction uses the fixed
forward and saved numeric tensors; update programs never execute on selection or
test examples. The score is0.9*correct+0.1*p_true. Crossover is disabled.

## Final held-out results

'''+ '\n'.join(table)+'''

![Three-seed accuracy](figures/rich/accuracy.png)

Image final splits contain10000 examples each; SST-2 uses its official872-example
labeled validation split as final test. Every model selection was frozen before
final inference. Three seeds are exploratory. Conditional paired intervals and
exact McNemar values are stored in rich_runs/rich_main/FINAL_RESULTS.json; repeated
use of the same final split does not create independent datasets.

## What this comparison establishes

Differences compare repeated LLM reflection plus GEPA selection against a strong,
feedback-assisted one-shot constructor. They do not isolate Pareto selection from
repeated reflection or additional proposals; a greedy-reflection control is needed.
Closed-form fitting can itself explain much of the one-shot performance. Inspect
the actual programs rather than attributing all accuracy to direct numerical
reasoning by the language model. Parameter sparsity and changed-weight counts are
saved alongside the results; a transformer with zeroed attention branches does not
establish that its attention learned the task.

## Information boundary and provenance

Rich feedback is training data. Programs can calculate statistics or fit weights
from it. Selection inputs, labels and trajectories stay private, and the final test
stays sealed until selection is complete. Bounded no-network execution exposes only
NumPy/math, current weights and the feedback snapshot. Full matrices remain in the
sandbox; prompt values are rounded and summarized where needed. Programs can return
further statistics for later reflection. Snapshot hashes, array schemas, parent
checkpoints, program sources and immutable dataset manifests permit reconstruction.
Extra harness feedback forwards and two deterministic program executions per edit
are recorded separately. Public-data familiarity in the LLM pretraining cannot be
ruled out.

Earlier Python experiments and partial JSON-only strict runs used different
protocols and are retained separately. They are not pooled into these tables.
The rich pilots rejected a harmless NumPy import; after allowing NumPy/math imports,
identical saved responses were replayed without another LLM call. Primary prompts
and sources were frozen after this interface check. All failures are retained.

See RICH_PROTOCOL.md and runtime/RICH_AUDIT.json. The audit verifies feedback-example
provenance, selection-trace boundaries, credentials and lists program operations for
mechanism review. Statistical fitting is permitted; optimizer absence needs code
review beyond checking API names. Diffusion was optional and is not included here.

Motivation: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [GEPA](https://arxiv.org/abs/2507.19457). This tests a combination on different
models and tasks rather than reproducing their headline claims.
'''
 blog+=f'\nWhole-project accounting: {len(ledger["calls"])} requests, ${ledger["spent"]:.4f} conservatively accounted, including all archived runs and pilots. Unknown-billing requests charged at their entire reserved upper bound: {unknown}. Original ceilings: $20 and650 requests.\n'
 (ROOT/'BLOG_RICH.md').write_text(blog)
 if (ROOT/'BLOG.md').exists() and not (ROOT/'BLOG_PREVIOUS_PROTOCOL.md').exists():(ROOT/'BLOG.md').rename(ROOT/'BLOG_PREVIOUS_PROTOCOL.md')
 (ROOT/'BLOG.md').write_text(blog)
if __name__=='__main__':torch.set_num_threads(2);finalize()
