"""Assemble all completed primary tasks without additional model selection or API calls."""
import json,time
from pathlib import Path
import numpy as np
import experiment as common
import text_experiment as text
ROOT=common.ROOT

def assemble():
 image=json.loads((ROOT/'runs/main/FINAL_RESULTS.json').read_text())
 language=json.loads((ROOT/'text_runs/FINAL_RESULTS.json').read_text())
 rows=[]
 for task in ['mnist','fashion','sst2']:
  for seed in [0,1,2]:
   folder=(ROOT/'text_runs/sst2' if task=='sst2' else ROOT/'runs/main'/task)/f'seed{seed}'
   initial=text.initial(task,seed) if task=='sst2' else common.initial(task,seed)
   construction=common.load_npz(folder/'one_shot.npz');best=common.load_npz(folder/'gepa_best.npz')
   record={'task':task,'seed':seed,'total_parameters':sum(v.size for v in best.values()),
           'nonzero_parameters':sum(int(np.count_nonzero(v)) for v in best.values()),
           'changed_from_initial':sum(int(np.count_nonzero(v!=initial[k])) for k,v in best.items()),
           'changed_from_construction':sum(int(np.count_nonzero(v!=construction[k])) for k,v in best.items()),
           'construction_status':json.loads((folder/'construction_status.json').read_text())}
   if task=='sst2':
    record['zero_output_branches']={b+'.'+branch:all(np.count_nonzero(best[b+'.'+branch+'.'+p])==0 for p in ['weight','bias']) for b in ['block0','block1'] for branch in ['proj','ff2']}
   rows.append(record)
 common.dump(ROOT/'PARAMETER_AUDIT.json',rows)
 table=['| Task | Seed | Total weights | Nonzero | Changed from initialization | Changed by GEPA |','|---|---:|---:|---:|---:|---:|']
 for r in rows:table.append(f'| {r["task"]} | {r["seed"]} | {r["total_parameters"]:,} | {r["nonzero_parameters"]:,} | {r["changed_from_initial"]:,} | {r["changed_from_construction"]:,} |')
 text_part=(ROOT/'BLOG_TRANSFORMER.md').read_text()
 blog=(ROOT/'BLOG.md').read_text()+'\n\n'+text_part+'\n\n## What the nominal model size hides\n\n'+ '\n'.join(table)+'\n\nNonzero counts include retained random weights; they do not establish learned features. Changed-weight counts measure edits, not the number of independently chosen coefficients: formulas can tie many values. `PARAMETER_AUDIT.json` also identifies transformer attention and feed-forward branches whose output weights and biases are entirely zero. Such branches contribute nothing even though the architecture still contains them.\n'
 blog+='\n## Scope of the evidence\n\nThese comparisons test one-shot construction against LLM reflection plus GEPA selection and against Adam on three architectures. They do not isolate the benefit of Pareto selection from repeated reflection, extra feedback, or additional proposals. A greedy-reflection control and matched-example one-shot control remain necessary for that attribution. The earlier numerical-fitting runs are a separate broader method, not test leakage by definition, and are excluded from these strict tables. Diffusion was planned as optional exploratory work and was not run in this batch.\n'
 ledger=json.loads((ROOT/'runtime/usage.json').read_text());blog+=f'\nWhole-project ledger at assembly: {len(ledger["calls"])} calls, ${ledger["spent"]:.4f} estimated token cost, including all pilots and archived experiments. The original $20/650-call whole-project ceilings cover all phases; the initial image-only conservative cap was $10.\n'
 (ROOT/'BLOG_COMPLETE.md').write_text(blog)
 common.dump(ROOT/'runtime/ALL_RESULTS_READY.json',{'time':time.time(),'blog':str(ROOT/'BLOG_COMPLETE.md'),'image_results':image['aggregate'],'text_results':language['aggregate'],'cost':ledger['spent']})
 print('ALL_RESULTS_READY: BLOG_COMPLETE.md, three architecture suites and parameter audit',flush=True)
if __name__=='__main__':
 while not all((ROOT/'runtime'/n).exists() for n in ['RESULTS_READY.json','TEXT_RESULTS_READY.json']):
  if (ROOT/'runtime/ATTENTION.json').exists():raise RuntimeError('Image supervisor needs attention')
  ledger=json.loads((ROOT/'runtime/usage.json').read_text())
  if ledger['blocked']:raise RuntimeError('Global budget or service stop')
  time.sleep(30)
 assemble()
