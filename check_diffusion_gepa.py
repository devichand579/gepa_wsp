"""Real GEPA acceptance with feedback-only statistical code and numeric checkpoints."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
import experiment as common
import rich_experiment as rich
import diffusion_core as core
import diffusion_experiment as experiment

def check():
 torch.set_num_threads(2)
 state=core.initial(0);clean=np.zeros((4,1,16,16),dtype=np.float32);x,t,y=core.fixed_batch(clean,42,'cpu');data={'clean':clean,'x':x.numpy(),'t':t.numpy(),'y':y.numpy()}
 measured,_=experiment.forward(state,data,np.arange(4),'cpu',capture=True,features=True);expected=core.evaluate(state,clean,42,'cpu')
 assert abs(measured['mse']-expected['mse'])<1e-6 and measured['features'].shape==(4,64,16,16)
 with tempfile.TemporaryDirectory() as td:
  folder=Path(td);state={k:np.zeros_like(v) for k,v in state.items()};state['out.bias'][:]=1
  compiler=rich.Compiler(experiment.BACKEND,'diffusion',state,folder/'checkpoints');candidate={m:json.dumps({'checkpoint':'base'}) for m in experiment.MODULES}
  train={'clean':np.zeros((8,1,16,16),dtype=np.float32),'x':np.zeros((8,1,16,16),dtype=np.float32),'t':np.zeros(8,dtype=np.int64),'y':np.zeros((8,1,16,16),dtype=np.float32)}
  selection={k:v.copy() for k,v in train.items()};selection['x'][:]=.5
  adapter=experiment.Adapter(0,'cpu',folder,compiler,{'feedback':train,'selection':selection})
  try:adapter.evaluate([{'split':'selection','index':0}],candidate,True)
  except AssertionError:pass
  else:raise AssertionError('Selection trace exposed')
  code='''def edit(w,f):
    assert np.all(f["x"]==0), "Selection inputs leaked into fitting"
    w["out.bias"][:]=np.mean(f["y"])
    return {"weights":w,"diagnostics":{"training_mean":float(f["y"].mean())}}
'''
  with patch.object(experiment.client,'call',return_value=(code,'Toy statistical update.\n```python\n'+code+'```')),patch.object(experiment.rich,'execute',wraps=rich.execute) as execution:
   result=gepa.optimize(seed_candidate=candidate,trainset=[{'split':'feedback','index':i} for i in range(8)],valset=[{'split':'selection','index':i} for i in range(8)],adapter=adapter,module_selector='all',reflection_minibatch_size=8,skip_perfect_score=False,stop_callbacks=[MaxCandidateProposalsStopper(1)],max_metric_calls=100,run_dir=str(folder/'gepa'))
   assert execution.call_count==1,'Weight update re-executed during private selection evaluation'
  assert result.num_candidates==2 and max(result.val_aggregate_scores)==0
  rows=[json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]
  assert all(not r['capture_traces'] for r in rows if r['split']=='selection')
  manifest=json.loads((folder/'llm/mutation_001/feedback_manifest.json').read_text());assert manifest['split']=='feedback' and manifest['n_full_feedback']==8
 print('PASS: diffusion evaluator matches baseline; full hidden features; real GEPA acceptance; training-only fitting; selection never executes update code; no paid API calls')
if __name__=='__main__':check()
