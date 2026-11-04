"""Evaluate the predeclared expanded matrix only after every selection is frozen.

No LLM calls, fitting or model selection occur here. Cached predictions make
restarts inexpensive; all published models come from the immutable prefix files.
"""
from __future__ import annotations
import ast,csv,fcntl,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import binomtest
import experiment as common
import rich_experiment as rich
import diffusion_core as core
import expansion_queue as queue
ROOT=common.ROOT
OUT=ROOT/'expanded_results'

def validate_gate():
 path=ROOT/'runtime/EXPANSION_SELECTIONS_FROZEN.json'
 assert path.exists(),'Final-test access locked: expanded selections are incomplete'
 frozen=json.loads(path.read_text());plan=json.loads((ROOT/'runtime/EXPANSION_PLAN.json').read_text())
 assert frozen['plan_sha256']==plan['plan_sha256']==hashlib.sha256((ROOT/plan['plan']).read_bytes()).hexdigest(),'Plan changed after selection freeze'
 sources=json.loads((ROOT/'runtime/EXPANSION_FROZEN.json').read_text())['source_hashes']
 for name,sha in sources.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Frozen source changed: '+name
 configuration=json.loads((ROOT/'runtime/EXPANSION_FROZEN.json').read_text())
 if 'repair_manifest_sha256' in configuration:assert hashlib.sha256((ROOT/'runtime/EXPERIMENT_REPAIRS.json').read_bytes()).hexdigest()==configuration['repair_manifest_sha256'],'Repair mapping changed after freeze'
 expected={job['id']:job for group in ['classification','diffusion'] for job in queue.jobs(group)}
 assert set(frozen['selections'])==set(expected),'Incomplete or unexpected experiment matrix'
 for name,job in expected.items():
  assert queue.verify_job(job),name
  for budget in job['prefixes']:
   record=json.loads((job['folder']/'prefixes'/f'budget{budget:03d}.json').read_text())
   assert record==frozen['selections'][name][str(budget)],'Selection altered after freeze: '+name
 return list(expected.values())

def selected_programs(job):
 """Trace numeric checkpoint parents back to successful weight-update programs."""
 folder=job['folder'];index={};selected=set();visited=set()
 status=json.loads((folder/'construction_status.json').read_text())
 if status['error'] is None and (folder/'construction.py').exists():selected.add(hashlib.sha256((folder/'construction.py').read_bytes()).hexdigest())
 for path in (folder/'llm').glob('mutation_*/edit.py'):
  manifest=path.parent/'feedback_manifest.json'
  if not manifest.exists() or not (path.parent/'program_diagnostics.json').exists():continue
  record=json.loads(manifest.read_text());source=path.read_text();parent=record['parent_checkpoint']
  child=common.digest({'parent':parent,'source':source,'feedback':record['arrays_sha256']})
  index[child]=(hashlib.sha256(source.encode()).hexdigest(),parent)
 def walk(candidate):
  if candidate in visited:return
  visited.add(candidate);path=folder/'checkpoints'/f'{candidate}.json';assert path.exists(),'Missing selected lineage: '+candidate
  components=json.loads(path.read_text())
  for entry in components.values():
   checkpoint=json.loads(entry)['checkpoint']
   if checkpoint=='base':continue
   assert checkpoint in index,'Missing update provenance: '+checkpoint
   sha,parent=index[checkpoint];selected.add(sha);walk(parent)
 for budget in job['prefixes']:walk(json.loads((folder/'prefixes'/f'budget{budget:03d}.json').read_text())['candidate'])
 return selected

def audit(jobs):
 secret=Path('/tmp/gepa-weight-space-fireworks.key').read_text().strip();checked=0;programs={};selected=set()
 for job in jobs:
  selected|=selected_programs(job)
  folder=job['folder'];diffusion='diffusion' in job['id'];seed=int(folder.name.removeprefix('seed'));task='diffusion' if diffusion else folder.parent.name
  data_folder=ROOT/'data'/('diffusion' if diffusion else task)/f'seed{seed}'
  feedback=common.load_npz(data_folder/'feedback.npz');n=len(feedback['clean'] if diffusion else feedback['y'])
  if not diffusion:b=rich.backend(task,seed)
  for path in (folder/'llm').rglob('request.json'):
   request=json.loads(path.read_text());raw=path.read_text();assert secret not in raw
   user=request['messages'][1]['content'];assert [m['role'] for m in request['messages']]==['system','user']
   assert 'selection_index' not in user and 'test_index' not in user
   assert '"split": "selection"' not in user and '"split": "test"' not in user
   payload=json.JSONDecoder().raw_decode(user[user.index('{"network":'):])[0]
   for ex in payload['feedback']['examples']:
    i=ex['feedback_index'];assert 0<=i<n
    if not diffusion:
     assert ex['label']==int(feedback['y'][i])
     if 'image_28x28_ascii' in ex:assert ex['image_28x28_ascii']==common.ascii_image(feedback['x'][i])
     if 'tokenized_text' in ex:
      import text_experiment
      assert ex['tokenized_text']==text_experiment.decode(feedback['x'][i])
   manifest=json.loads((path.parent/'feedback_manifest.json').read_text());assert manifest['split']=='feedback' and manifest['n_full_feedback']==n
   assert all(0<=i<n for i in manifest['batch_indices']);checked+=1
  for line in (folder/'evaluations.jsonl').read_text().splitlines():
   row=json.loads(line)
   if row['split']=='selection':assert not row['capture_traces']
  for path in [folder/'construction.py',*(folder/'llm').rglob('edit.py')]:
   if not path.exists():continue
   source=path.read_text();assert secret not in source;sha=hashlib.sha256(source.encode()).hexdigest()
   if sha in programs:continue
   try:tree=ast.parse(source)
   except SyntaxError:
    assert sha not in selected,'Invalid syntax in selected program: '+str(path)
    programs[sha]={'source':str(path.relative_to(ROOT)),'calls':[],'for_iterators':[],'while_loops':0,'review_flags':['Rejected source has invalid Python syntax']};continue
   calls=sorted({ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)})
   names={node.id.lower() for node in ast.walk(tree) if isinstance(node,ast.Name)}
   flags=[name for name in calls if any(term in name.lower() for term in ['backward','adam','sgd','autograd','gradient'])]
   suspicious={name for name in names if any(term in name for term in ['grad','hess','newton','deriv','learning_rate','optimizer']) or name in {'epochs','num_epochs','g_w','g_b','gw','gb','dce','dce0','d2ce','d2ce0'}}
   if suspicious:flags.append('Potential manual optimizer variable names: '+str(sorted(suspicious)))
   if any(term in source.lower() for term in ['newton step','gradient-descent','gradient descent','gradient of','upstream gradient']):flags.append('Optimizer-related source text requires inspection; comments and unused helpers alone do not establish a violation')
   programs[sha]={'source':str(path.relative_to(ROOT)),'calls':calls,'for_iterators':[ast.unparse(node.iter) for node in ast.walk(tree) if isinstance(node,ast.For)],'while_loops':sum(isinstance(node,ast.While) for node in ast.walk(tree)),'review_flags':flags}
 for sha,record in programs.items():record.update(source_sha256=sha,in_selected_lineage=sha in selected)
 result={'passed_provenance_checks':True,'requests_checked_including_preserved_continuation_copies':checked,'programs':list(programs.values()),'review_flags':sum(bool(p['review_flags']) for p in programs.values()),'selected_review_flags':sum(bool(p['review_flags']) and p['in_selected_lineage'] for p in programs.values()),'mechanism_review':'Statistics and training-only model selection are allowed. AST summaries do not prove absence of a manually written gradient optimizer; inspect flagged programs and selected lineages.'}
 common.dump(OUT/'audit.json',result);return result

def require_mechanism_review(checked):
 path=ROOT/'runtime/EXPANSION_MECHANISM_REVIEW.json';decisions=json.loads(path.read_text()) if path.exists() else {}
 unresolved=[]
 for program in checked['programs']:
  decision=decisions.get(program['source_sha256'],{})
  if program['in_selected_lineage']:
   if decision.get('verdict')!='allowed_statistics':unresolved.append(program)
 if unresolved:
  common.dump(ROOT/'runtime/EXPANSION_MECHANISM_REVIEW_REQUIRED.json',{'time':time.time(),'programs':unresolved,'note':'Every selected source requires a recorded source review, including unflagged code. No model may be replaced using final-test results.'})
  raise RuntimeError('Selected program mechanism review required before final-test access')

def paired_classification(base,current,y):
 a=base==y;b=current==y;d=b.astype(float)-a;se=float(d.std(ddof=1)/math.sqrt(len(d)));gained=int((~a&b).sum());lost=int((a&~b).sum())
 return {'paired_gain':float(d.mean()),'conditional_normal_95ci':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],'wrong_to_right':gained,'right_to_wrong':lost,'mcnemar_exact_p':float(binomtest(gained,gained+lost,.5).pvalue) if gained+lost else 1.}

def charged_cost(folder,budget,ledger):
 from classification_followup import dispatches
 ids={d['ledger_call'] for d in dispatches(folder)[:budget]}
 construction=folder/'llm/construction/dispatch.json'
 if construction.exists():ids.add(json.loads(construction.read_text())['ledger_call'])
 return sum(ledger['calls'][i].get('cost',0) for i in ids),ids

def classification_eval(task,seed,weights,data):
 state=common.load_npz(weights);digest=common.digest({'weights':rich.arrays_digest(state),'test_arrays':rich.arrays_digest(data)});cache=OUT/'cache'/f'class_{task}_{seed}_{digest}';cache.parent.mkdir(exist_ok=True)
 if cache.with_suffix('.json').exists():
  meta=json.loads(cache.with_suffix('.json').read_text());pred=common.load_npz(cache.with_suffix('.npz'))['pred']
 else:
  b=rich.backend(task,seed);result,_=rich.forward(b,task,state,data['x'],data['y'],'cuda:0')
  pred=result['pred'];meta=common.summary(result);np.savez(cache.with_suffix('.npz'),pred=pred);common.dump(cache.with_suffix('.json'),meta)
 counts={'parameter_count':sum(v.size for v in state.values()),'nonzero_parameters':sum(int(np.count_nonzero(v)) for v in state.values())}
 if task=='sst2':counts['zero_output_branches']={block+'.'+branch:all(np.count_nonzero(state[block+'.'+branch+'.'+p])==0 for p in ['weight','bias']) for block in ['block0','block1'] for branch in ['proj','ff2']}
 return {**meta,**counts,'n_test':len(data['y'])},pred

def generation(state,seed,training):
 digest=rich.arrays_digest(state);base=OUT/'generations'/f'seed{seed}_{digest}';base.parent.mkdir(exist_ok=True)
 if base.with_suffix('.json').exists():return json.loads(base.with_suffix('.json').read_text())
 images=core.sample(state,128,89000+seed,'cuda:1');np.savez_compressed(base.with_suffix('.npz'),images=images)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(8,8,figsize=(7,7))
 for ax,image in zip(axes.ravel(),images[:64]):ax.imshow(image[0],cmap='gray',vmin=-1,vmax=1);ax.axis('off')
 fig.subplots_adjust(wspace=.04,hspace=.04);fig.savefig(base.with_suffix('.png'),dpi=150,bbox_inches='tight');plt.close(fig)
 query=images.reshape(len(images),-1);reference=training.reshape(len(training),-1)
 distances=np.maximum((query*query).sum(1)[:,None]+(reference*reference).sum(1)[None,:]-2*query@reference.T,0)/256
 report={'n_samples':128,'sample_seed':89000+seed,'grid':str(base.with_suffix('.png').relative_to(ROOT)),'samples':str(base.with_suffix('.npz').relative_to(ROOT)),'pixel_mean':float(images.mean()),'pixel_std':float(images.std()),'spatial_total_variation':float((np.abs(np.diff(images,axis=2)).mean()+np.abs(np.diff(images,axis=3)).mean())/2),'mean_nearest_training_pixel_mse':float(distances.min(1).mean()),'note':'Pixel statistics and training-neighbor distances are descriptive, not validated perceptual quality scores.'}
 common.dump(base.with_suffix('.json'),report);return report

def diffusion_eval(seed,weights,clean,training):
 state=common.load_npz(weights);digest=common.digest({'weights':rich.arrays_digest(state),'test_clean_sha256':hashlib.sha256(clean.tobytes()).hexdigest()});base=OUT/'cache'/f'diffusion_{seed}_{digest}';base.parent.mkdir(exist_ok=True)
 if base.with_suffix('.json').exists():meta=json.loads(base.with_suffix('.json').read_text());loss=common.load_npz(base.with_suffix('.npz'))['mse']
 else:
  result=core.evaluate(state,clean,83000+seed,'cuda:1');loss=result['per_example_mse'];meta={'mse':result['mse'],'score':result['score']};np.savez(base.with_suffix('.npz'),mse=loss);common.dump(base.with_suffix('.json'),meta)
 return {**meta,'n_test':len(clean),'generation':generation(state,seed,training)},loss

def aggregate(rows,metric):
 result=[];keys=sorted({(r['group'],r['task'],r['model'],r['budget']) for r in rows})
 for group,task,model,budget in keys:
  values=[r for r in rows if (r['group'],r['task'],r['model'],r['budget'])==(group,task,model,budget)]
  result.append({'group':group,'task':task,'model':model,'budget':budget,'n_seeds':len(values),metric+'_mean':float(np.mean([v[metric] for v in values])),metric+'_sd':float(np.std([v[metric] for v in values],ddof=1)) if len(values)>1 else 0.,'mean_cumulative_api_cost':float(np.mean([v.get('cumulative_api_cost',0) for v in values]))})
 return result

def proposal_reliability(folder,budget):
 """Count full-trajectory submission outcomes, including unsuccessful API slots."""
 status=json.loads((folder/'construction_status.json').read_text())
 successes={int(p.parent.name.removeprefix('mutation_')) for p in (folder/'llm').glob('mutation_*/program_diagnostics.json')}
 errors={}
 path=folder/'proposal_errors.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   row=json.loads(line);errors[int(row['proposal'])]=row['error']
 counts={}
 for number,error in errors.items():
  if number in successes:continue
  if 'METHOD_VIOLATION' in error:kind='method_rejection'
  elif 'Incomplete LLM response: length' in error:kind='response_truncation'
  elif any(s in error for s in ['Provider failure','Provider rejected','Missing usage','HTTPError']):kind='provider_failure'
  elif any(s in error for s in ['MemoryError','timed out','CPU time limit','Killed']):kind='resource_limit'
  elif any(s in error for s in ['ValueError','TypeError','IndexError','AssertionError','SyntaxError','IndentationError','LinAlgError','NameError']):kind='code_or_array_error'
  else:kind='other_failure'
  counts[kind]=counts.get(kind,0)+1
 return {'construction_succeeded':status['error'] is None,'mutation_budget':budget,'valid_mutation_programs':len(successes),'failed_mutation_proposals':sum(counts.values()),'failure_counts':counts,'unclassified_slots':max(0,budget-len(successes)-sum(counts.values())),'note':'Execution validity is distinct from GEPA acceptance or test improvement. Counts cover the complete trajectory, not each nested prefix.'}

def make_report(result):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 figures=ROOT/'figures/expanded';figures.mkdir(parents=True,exist_ok=True)
 for group,title in [('budget','DeepSeek budget study: original representation'),('models','Matched compact-prompt model comparison')]:
  fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
  for ax,task in zip(axes,['mnist','fashion','sst2']):
   for model in ['deepseek','minimax','nemotron']:
    rows=sorted([r for r in result['classification_summary'] if r['group']==group and r['task']==task and r['model']==model],key=lambda r:r['budget'])
    if rows:ax.errorbar([r['budget'] for r in rows],[r['accuracy_mean']*100 for r in rows],yerr=[r['accuracy_sd']*100 for r in rows],marker='o',capsize=3,label=model)
   baseline=np.mean([r['accuracy'] for r in result['baselines'] if r['task']==task]);ax.axhline(baseline*100,color='black',linestyle='--',label='Adam');ax.set_title(task);ax.set_xlabel('Mutation API calls');ax.set_ylim(0,100);ax.legend(fontsize=8)
  axes[0].set_ylabel('Final-test accuracy (%)');fig.suptitle(title);fig.tight_layout();fig.savefig(figures/(group+'.png'),dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4))
 for model in ['deepseek','minimax','nemotron']:
  rows=sorted([r for r in result['diffusion_summary'] if r['model']==model],key=lambda r:r['budget']);ax.errorbar([r['budget'] for r in rows],[r['mse_mean'] for r in rows],yerr=[r['mse_sd'] for r in rows],marker='o',capsize=3,label=model)
 ax.axhline(np.mean([r['mse'] for r in result['baselines'] if r['task']=='diffusion']),color='black',linestyle='--',label='Adam');ax.set_xlabel('Mutation API calls');ax.set_ylabel('Final denoising MSE (lower is better)');ax.legend();fig.tight_layout();fig.savefig(figures/'diffusion.png',dpi=180);plt.close(fig)
 table=['| Study | Task | Model | Calls | Mean final result ± seed SD | Mean cumulative API cost |','|---|---|---|---:|---:|---:|']
 for r in result['classification_summary']:table.append(f'| {r["group"]} | {r["task"]} | {r["model"]} | {r["budget"]} | {100*r["accuracy_mean"]:.2f}% ± {100*r["accuracy_sd"]:.2f} pp | ${r["mean_cumulative_api_cost"]:.4f} |')
 for r in result['diffusion_summary']:table.append(f'| diffusion | MNIST16 | {r["model"]} | {r["budget"]} | MSE {r["mse_mean"]:.5f} ± {r["mse_sd"]:.5f} | ${r["mean_cumulative_api_cost"]:.4f} |')
 text='''# Reflective weight programming across budgets, LLMs and diffusion

The LLM writes feedback-aware NumPy weight-update programs for fixed neural
networks. Full weights and permitted training arrays are available to each program;
statistics, covariance/PCA and closed-form fitting are allowed. GEPA supplies paired
feedback evaluation and Pareto candidate selection. This measures the whole
reflection-and-selection strategy around a generated statistical program. It does
not isolate GEPA's Pareto mechanism or show that coefficients were chosen without
numerical algorithms. See the generated-program audit before claiming optimizer
absence.

## Predeclared comparisons

The original-representation DeepSeek study uses60 mutation calls, with
prefixes0,5,10,30,60. Eligible original trajectories continue from30 calls.
Fresh matched compact-prompt runs compare DeepSeek,
MiniMax M3.1 Flash and Nemotron Lightning3.5 at0,5,10 calls. All use seeds0,1,2.
MNIST uses a101770-parameter MLP; Fashion-MNIST a20490-parameter CNN; SST-2 an
83266-parameter transformer. The diffusion target is a56609-parameter convolutional
epsilon predictor with a fixed32-step cosine DDPM on MNIST16x16. Its three LLMs
are compared at0,5,10,20 calls, with Adam trained separately.

Before final-test evaluation, source review found prohibited explicit parameter
gradient/Newton updates in original DeepSeek MNIST seeds0/1/2, Fashion-MNIST
seeds1/2 and Nemotron diffusion seeds0/1/2. Those full trajectories were excluded and replaced by fresh
constructions and searches with the same models, data, seeds and full budgets.
Original artifacts and costs are retained. The replacements use a pre-execution
method screen, followed by selected-lineage review; a static screen alone cannot
prove that every possible optimizer implementation is absent. The replacement
mapping is recorded in runtime/EXPERIMENT_REPAIRS.json. No test results informed
this repair. The affected classification budget prefixes belong to fresh60-call runs.

Parameter counts describe the fixed network containers, not how many parameters
were usefully fitted. Source inspection found SST-2 programs that zero attention
and feedforward blocks and encode a bag-of-words ridge classifier in embeddings
and the head. Other programs fit only a readout over fixed or constructed
features. Such results demonstrate compilation of statistical procedures into
network weights; they do not by themselves establish gradient-free learning of
attention or every hidden layer. The proposal budgets count LLM requests, including
failed submissions. Each generated program may perform many training-only fits,
augmentations and forward passes within a60-CPU-second/4-GiB sandbox limit; the
program is executed twice to check determinism. Equal request counts do not imply
equal FLOPs, data exposure or parameter-update counts. The Adam comparison does
not match this internal computation. These statistical fits still constitute
learning from data; the optimizer restriction excludes parameter-gradient methods,
not all numerical fitting algorithms.

Both one-shot and GEPA have the same initial access to all permitted training
arrays and initial forward outputs. GEPA starts from that construction and gets
additional reflection/evaluation calls. Classification selects by0.9*correct+
0.1*p_true; diffusion by negative MSE. Selection data influences search but never
enters reflection or generated code. Every expanded selection was frozen before
final-test access. Budget prefixes exclude later candidates and follow the SAME
trajectory; they are not independent runs. More budget need not improve test
accuracy, and the composite selection objective need not yield monotone accuracy.

## Final held-out results

'''+ '\n'.join(table)+'''

![Budget study](figures/expanded/budget.png)

![Matched model study](figures/expanded/models.png)

![Diffusion denoising](figures/expanded/diffusion.png)

Three seeds are exploratory. Dots/error bars summarize seed variation, not all
LLM sampling uncertainty. Conditional paired test-example intervals and unadjusted
McNemar p-values are in expanded_results/results.json. The same final split is
reused across seeds; these are not independent test datasets. Image final splits
have10000 examples; SST-2 uses its872-example official labeled validation split.
A greedy-reflection control would be needed to attribute improvements specifically
to Pareto selection.

## Diffusion samples

Final noise draws and generation seeds differ from reflection feedback. Adam
refreshes training noise each optimizer step; the LLM conditions use their fixed
paired training noise bank. All share the same clean-image pool. This is not an
equal-compute or identical-noise-exposure comparison. Lower denoising MSE alone
is not evidence of useful generation. All fixed-seed sample arrays/grids are saved;
these pixel summaries are not FID or validated perceptual metrics.

'''
 for row in result['diffusion']:
  if row['seed']==0 and row['budget'] in [0,20]:text+=f'\n{row["model"]}, seed0, budget{row["budget"]}:\n\n![Fixed first64 draws]({row["generation"]["grid"]})\n'
 baseline=next(r for r in result['baselines'] if r['task']=='diffusion' and r['seed']==0);text+=f'\nAdam, seed0:\n\n![Adam fixed first64 draws]({baseline["generation"]["grid"]})\n'
 text+='''
## Reliability, mechanism and cost

Construction failures retain random initialization; their gains include recovery
from a failed submission. Provider errors and invalid programs consume call slots.
Context-rejected compatibility runs are retained separately and included in whole-
project accounting. Compact display prompts solve the smaller model's context
limit; full-precision arrays inside edit(w,f) are unchanged. Original DeepSeek
budget trajectories retain their original representation, while the separate
fresh model comparison uses the same compact representation for all three LLMs.

The audit checks feedback-example provenance, selection trace boundaries, source
and checkpoint hashes, and credential artifacts, and lists generated-program calls
and loops for mechanism review. Absence of a named optimizer API is not proof that
no manual gradient loop was written. Retained random features, zeroed attention
branches and nonzero parameter counts also need careful interpretation: nominal
architecture alone does not establish learned attention or all-layer learning.
Public-dataset familiarity in the LLM pretraining cannot be ruled out.

Cumulative cost rows reuse earlier calls; do not sum costs across nested budgets.
Whole-project accounting includes every historical pilot and discarded condition.
Unknown billing is charged at its full reserved upper bound. No new paid inference
is performed in this evaluator. See EXPANSION_PLAN.md, runtime/EXPANSION_FROZEN.json,
expanded_results/audit.json and the preserved requests/programs/checkpoints.

Motivation: [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [GEPA](https://arxiv.org/abs/2507.19457). These experiments test a different
combination rather than reproduce their headline results.
'''
 reliability=result.get('reliability',[])
 if reliability:
  groups={}
  for row in reliability:
   key=(row['group'],row['task'],row['model']);summary=groups.setdefault(key,{'valid':0,'budget':0,'construction_failures':0,'failures':{}})
   summary['valid']+=row['valid_mutation_programs'];summary['budget']+=row['mutation_budget'];summary['construction_failures']+=int(not row['construction_succeeded'])
   for kind,count in row['failure_counts'].items():summary['failures'][kind]=summary['failures'].get(kind,0)+count
  text+='\n### Proposal reliability\n\nThese counts pool the full three-seed trajectories once, rather than counting nested budget prefixes repeatedly. A valid program need not be accepted or improve a score.\n\n| Study | Task | LLM | Valid mutation programs / slots | Failed one-shot constructions | Mutation failure breakdown |\n|---|---|---|---:|---:|---|\n'
  for (group,task,model),summary in sorted(groups.items()):
   breakdown=', '.join(f'{kind}: {count}' for kind,count in sorted(summary['failures'].items())) or 'none'
   text+=f'| {group} | {task} | {model} | {summary["valid"]}/{summary["budget"]} | {summary["construction_failures"]} | {breakdown} |\n'
 text+=f'\nConservatively accounted project API cost: ${result["accounting"]["whole_project_cost"]:.4f} / $50; requests: {result["accounting"]["whole_project_requests"]}. Retained program audit flags, including rejected proposals: {result["audit_review_flags"]}. Every selected source passed recorded inspection before final tests.\n'
 (ROOT/'BLOG_EXPANDED.md').write_text(text)
 with (OUT/'summary.csv').open('w',newline='') as f:
  keys=['group','task','model','budget','n_seeds','accuracy_mean','accuracy_sd','mse_mean','mse_sd','mean_cumulative_api_cost'];writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(result['classification_summary']+result['diffusion_summary'])

def finalize():
 OUT.mkdir(exist_ok=True)
 with (OUT/'finalize.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (ROOT/'runtime/EXPANDED_RESULTS_READY.json').exists():return
  jobs=validate_gate();checked=audit(jobs);require_mechanism_review(checked)
  common.dump(ROOT/'runtime/EXPANSION_METHODS_REVIEWED.json',{'time':time.time(),'audit_sha256':hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest(),'selected_review_flags':checked['selected_review_flags'],'note':'Provenance verified; every selected source requires a recorded statistical-method decision, including unflagged programs. AST checks alone do not establish complete semantic optimizer absence.'})
  ledger=json.loads((ROOT/'runtime/usage.json').read_text());assert not any(c['status']=='pending' for c in ledger['calls'])
  assert ledger['spent']+ledger['reserved']<=50
  classification=[];diffusion=[];baselines=[];reliability=[];data_cache={};all_ids=set()
  for job in jobs:
   folder=job['folder'];seed=int(folder.name.removeprefix('seed'));is_diff='diffusion' in job['id'];task='diffusion' if is_diff else folder.parent.name
   model=job['args'][job['args'].index('--model')+1];group='diffusion' if is_diff else ('budget' if '--kind' in job['args'] and job['args'][job['args'].index('--kind')+1]=='continue' else 'models')
   reliability.append({'job':job['id'],'group':group,'task':task,'model':model,'seed':seed,**proposal_reliability(folder,job.get('budget',max(job['prefixes'])))})
   key=(task,seed)
   if key not in data_cache:
    if is_diff:
     with np.load(ROOT/'data/final_test/mnist.npz',allow_pickle=False) as z:x=z['x']
     clean=(F.interpolate(torch.from_numpy(x),size=(16,16),mode='bilinear',align_corners=False)*2-1).numpy();training=common.load_npz(ROOT/'data/diffusion'/f'seed{seed}/feedback.npz')['clean'];data_cache[key]=(clean,training)
    else:data_cache[key]=common.load_npz(ROOT/'data/final_test'/('sst2/seed'+str(seed)+'.npz' if task=='sst2' else task+'.npz'))
   if not any(r['task']==task and r['seed']==seed for r in baselines):
    if is_diff:
     clean,training=data_cache[key];base,_=diffusion_eval(seed,ROOT/'diffusion_runs/baselines'/f'seed{seed}/adam_best.npz',clean,training)
    else:base,_=classification_eval(task,seed,folder/'adam_best.npz',data_cache[key])
    baselines.append({'task':task,'seed':seed,'model':'adam',**base})
   zero=folder/'prefixes/budget000.npz'
   if is_diff:
    clean,training=data_cache[key];_,base_prediction=diffusion_eval(seed,zero,clean,training)
   else:_,base_prediction=classification_eval(task,seed,zero,data_cache[key])
   for budget in job['prefixes']:
    weights=folder/'prefixes'/f'budget{budget:03d}.npz';cost,ids=charged_cost(folder,budget,ledger);all_ids|=ids
    row={'group':group,'task':task,'model':model,'seed':seed,'budget':budget,'cumulative_api_cost':cost,'weights':str(weights.relative_to(ROOT)),'construction_error':json.loads((folder/'construction_status.json').read_text())['error']}
    if is_diff:
     metric,prediction=diffusion_eval(seed,weights,clean,training);difference=base_prediction-prediction;se=float(difference.std(ddof=1)/math.sqrt(len(difference)));row.update(metric,paired_mse_improvement=float(difference.mean()),conditional_normal_95ci=[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)]);diffusion.append(row)
    else:
     metric,prediction=classification_eval(task,seed,weights,data_cache[key]);row.update(metric,**paired_classification(base_prediction,prediction,data_cache[key]['y']));classification.append(row)
   common.dump(OUT/'evaluation_progress.json',{'completed_jobs':len({(r['group'],r['task'],r['model'],r['seed']) for r in classification+diffusion}),'total_jobs':len(jobs),'time':time.time()})
  study_cost=sum(ledger['calls'][i].get('cost',0) for i in all_ids)
  result={'classification':classification,'diffusion':diffusion,'baselines':baselines,'classification_summary':aggregate(classification,'accuracy'),'diffusion_summary':aggregate(diffusion,'mse'),'audit_review_flags':checked['review_flags'],'accounting':{'whole_project_cost':ledger['spent'],'whole_project_requests':len(ledger['calls']),'expanded_study_unique_dispatch_cost':study_cost,'historical_setup_and_compatibility_cost':ledger['spent']-study_cost,'cap':50},'notes':'All prefix selections frozen before tests. Nested budgets share trajectories. AST program audit requires semantic review before an optimizer-absence claim.'}
  result['reliability']=reliability
  common.dump(OUT/'results.json',result);make_report(result)
  common.dump(ROOT/'runtime/EXPANDED_RESULTS_READY.json',{'time':time.time(),'blog':str(ROOT/'BLOG_EXPANDED.md'),'results':str(OUT/'results.json'),'classification_rows':len(classification),'diffusion_rows':len(diffusion),'mechanism_review_flags':checked['review_flags'],'cost':ledger['spent']})
  print('EXPANDED_RESULTS_READY',flush=True)
if __name__=='__main__':torch.set_num_threads(2);finalize()
