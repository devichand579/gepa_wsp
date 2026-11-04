"""Exercise both final-evaluation branches using synthetic arrays and isolated output."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import expansion_finalize as report
import experiment as common

with tempfile.TemporaryDirectory() as td:
 root=Path(td);runtime=root/'runtime';runtime.mkdir();out=root/'expanded_results'
 (root/'data/final_test').mkdir(parents=True)
 np.savez(root/'data/final_test/mnist.npz',x=np.zeros((4,1,28,28),dtype=np.float32))
 calls=[{'status':'complete','cost':.1} for _ in range(4)];common.dump(runtime/'usage.json',{'calls':calls,'spent':.4,'reserved':0.,'cap':50})
 jobs=[]
 for task,model,parts in [('mnist','deepseek',['expansion_runs','classification','deepseek10','mnist','seed0']),('diffusion','nemotron',['diffusion_runs','models','nemotron','seed0'])]:
  folder=root.joinpath(*parts);(folder/'prefixes').mkdir(parents=True);(folder/'llm/construction').mkdir(parents=True);(folder/'llm/mutation_001').mkdir()
  common.dump(folder/'llm/construction/dispatch.json',{'ledger_call':2 if task=='mnist' else 3,'time':1})
  common.dump(folder/'llm/mutation_001/dispatch.json',{'ledger_call':0 if task=='mnist' else 1,'time':2});common.dump(folder/'construction_status.json',{'error':None})
  jobs.append({'id':model+'_'+task+'_0','folder':folder,'prefixes':[0,1],'args':['worker.py','--model',model,'--kind','models']})
 def fake_load(path):
  if str(path).endswith('feedback.npz'):return {'clean':np.zeros((2,1,16,16),dtype=np.float32)}
  return {'x':np.zeros((4,1,28,28),dtype=np.float32),'y':np.array([0,1,0,1])}
 def classification(task,seed,path,data):
  pred=np.array([0,1,0,1]) if '001' in str(path) else np.array([1,1,0,1])
  return {'accuracy':float((pred==data['y']).mean()),'cross_entropy':.2,'score':.9,'n_test':4},pred
 def diffusion(seed,path,clean,training):
  loss=np.full(4,.1 if '001' in str(path) else .2)
  return {'mse':float(loss.mean()),'score':-float(loss.mean()),'n_test':4,'generation':{'grid':'synthetic.png'}},loss
 audit={'programs':[],'review_flags':0,'selected_review_flags':0}
 def fake_audit(_):common.dump(out/'audit.json',audit);return audit
 with patch.object(report,'ROOT',root),patch.object(report,'OUT',out),patch.object(report,'validate_gate',return_value=jobs),patch.object(report,'audit',side_effect=fake_audit),patch.object(common,'load_npz',side_effect=fake_load),patch.object(report,'classification_eval',side_effect=classification),patch.object(report,'diffusion_eval',side_effect=diffusion),patch.object(report,'make_report') as blog,patch('builtins.print'):
  report.finalize();result=json.loads((out/'results.json').read_text());assert len(result['classification'])==2 and len(result['diffusion'])==2
  assert result['classification'][1]['paired_gain']==.25 and abs(result['diffusion'][1]['paired_mse_improvement']-.1)<1e-9
  assert result['accounting']['expanded_study_unique_dispatch_cost']==.4
  assert (runtime/'EXPANSION_METHODS_REVIEWED.json').exists() and (runtime/'EXPANDED_RESULTS_READY.json').exists() and blog.call_count==1
print('PASS: synthetic classification/diffusion finalization, paired gains, unique dispatch accounting, review/completion records; no real test arrays, API calls or weights used')
