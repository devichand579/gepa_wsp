"""Text backend checks, using synthetic labels rather than a final split."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
import text_experiment as t
import strict_experiment as s
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
import gepa

def check():
 t.install(0);state=t.initial('sst2',0);assert sum(v.size for v in state.values())==83266
 net=t.Network().eval();xx=torch.tensor([[2,3,0,0],[2,3,0,0]])
 assert torch.isfinite(net(xx)).all()
 assert torch.allclose(net(xx[:1]),net(torch.tensor([[2,3,0,0,0,0]])),atol=1e-6)
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);c=s.Compiler('sst2',state,td/'checkpoints')
  candidate={m:json.dumps({'parent':'base','program':{'ops':[]}}) for m in t.modules('sst2')}
  class TestAdapter(t.Adapter):
   def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
    self.proposals+=1;h,_=self.compiler.materialize(candidate)
    return {'head':json.dumps({'parent':h,'program':{'ops':[{'op':'set','tensor':'head.weight','value':0},{'op':'set','tensor':'head.bias','value':[5,0]}]}})}
  x=xx.numpy();y=np.zeros(2,dtype=np.int64)
  a=TestAdapter('sst2',0,'cpu',td,c,{'feedback':x,'selection':x},{'feedback':y,'selection':y})
  result=gepa.optimize(seed_candidate=candidate,trainset=[{'split':'feedback','index':i} for i in range(2)],valset=[{'split':'selection','index':i} for i in range(2)],adapter=a,module_selector=lambda *args:['head'],reflection_minibatch_size=2,skip_perfect_score=False,stop_callbacks=[MaxCandidateProposalsStopper(1)],max_metric_calls=20,run_dir=str(td/'gepa'))
  rows=[json.loads(line) for line in (td/'evaluations.jsonl').read_text().splitlines()]
  assert all(not row['capture_traces'] for row in rows if row['split']=='selection')
 print('PASS: transformer finite outputs, padding invariance, strict edit path, selection trace boundary')
if __name__=='__main__':torch.set_num_threads(2);check()
