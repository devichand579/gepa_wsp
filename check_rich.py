"""Validate richer forward feedback, fitting permission and evaluation isolation."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
import rich_experiment as rich
import experiment as common

def check():
 torch.set_num_threads(2)
 for task in ['mnist','fashion','sst2']:
  b=rich.backend(task,0);state=b.initial(task,0);data=common.load_npz(common.ROOT/'data'/task/'seed0/feedback.npz');x=data['x'][:4];y=data['y'][:4]
  actual,traces=rich.forward(b,task,state,x,y,'cpu',True)
  expected=b.evaluate_state(task,state,x,y,'cpu',True)
  assert np.allclose(actual['probs'],expected['probs'],atol=1e-6)
  assert np.allclose(actual['features'],expected['features'],atol=1e-6)
  assert traces and any('output' in k for k in traces)
 state=common.initial('mnist',0);features=np.zeros((4,128),dtype=np.float32);features[:2,0]=1;features[2:,1]=1
 feedback={'features':features,'y':np.array([0,0,1,1]),'x':np.zeros((4,1,28,28),dtype=np.float32)}
 code='''import numpy as np
from numpy.linalg import solve
def edit(w,f):
    H=f["features"].astype(np.float64)
    Y=np.eye(10)[f["y"]]
    w["fc2.weight"][:]=solve(H.T@H+np.eye(128),H.T@Y).T
    return {"weights":w,"diagnostics":{"class_counts":np.bincount(f["y"],minlength=10),"n":len(H)}}
'''
 result,report=rich.execute(state,feedback,code,['fc2.weight','fc2.bias'])
 assert np.all(result['fc2.weight'][:2,:2].diagonal()>0) and report['reported_statistics']['n']==4
 assert np.array_equal(result['fc1.weight'],state['fc1.weight'])
 for bad in ['def edit(w,f):\n    open("/tmp/gepa-weight-space-fireworks.key").read()\n    return w\n',
             'import os\ndef edit(w,f):\n    return w\n',
             'def edit(w,f):\n    np.load("/home/devichand/gepa-weight-space/data/final_test/mnist.npz")\n    return w\n',
             'def edit(w,f):\n    w["fc1.bias"][:]=2\n    return w\n']:
  try:rich.execute(state,feedback,bad,['fc2.weight','fc2.bias'])
  except (AssertionError,ValueError):pass
  else:raise AssertionError('Protected boundary bypassed')
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);state={k:np.zeros_like(v) for k,v in state.items()};state['fc2.bias'][1]=4
  compiler=rich.Compiler(common,'mnist',state,td/'checkpoints')
  candidate={m:json.dumps({'checkpoint':'base'}) for m in common.modules('mnist')}
  class Adapter(rich.Adapter):
   def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
    self.proposals+=1;assert self.latest['state'] is not None
    assert all(r['split']=='feedback' for r in self.latest_trajectories)
    updated={k:v.copy() for k,v in self.latest['state'].items()};updated['fc2.bias'][:]=0;updated['fc2.bias'][0]=4
    checkpoint='a'*64;np.savez(self.compiler.folder/f'{checkpoint}.npz',**updated)
    return {m:json.dumps({'checkpoint':checkpoint}) for m in components_to_update}
   def make_reflective_dataset(self,candidate,eval_batch,components):
    self.latest_trajectories=eval_batch.trajectories
    return super().make_reflective_dataset(candidate,eval_batch,components)
  x=np.zeros((8,1,28,28),dtype=np.float32);y=np.zeros(8,dtype=np.int64)
  adapter=Adapter(common,'mnist',0,'cpu',td,compiler,{'feedback':x,'selection':x},{'feedback':y,'selection':y})
  try:adapter.evaluate([{'split':'selection','index':0}],candidate,True)
  except AssertionError:pass
  else:raise AssertionError('Selection traces exposed')
  result=gepa.optimize(seed_candidate=candidate,trainset=[{'split':'feedback','index':i} for i in range(8)],valset=[{'split':'selection','index':i} for i in range(8)],adapter=adapter,module_selector='all',reflection_minibatch_size=8,skip_perfect_score=False,stop_callbacks=[MaxCandidateProposalsStopper(1)],max_metric_calls=100,run_dir=str(td/'gepa'))
  assert result.num_candidates==2 and max(result.val_aggregate_scores)>.9
  records=[json.loads(l) for l in (td/'evaluations.jsonl').read_text().splitlines()]
  assert all(not r['capture_traces'] for r in records if r['split']=='selection')
 print('PASS: all three fixed forwards match; complete layer traces; statistics/closed-form fitting allowed; file/import/module boundaries; real GEPA joint acceptance; selection gets frozen numeric checkpoints only')
if __name__=='__main__':check()
