import json,tempfile
from pathlib import Path
import numpy as np
import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
from strict_experiment import Compiler,Adapter,execute
import strict_experiment as strict
from unittest.mock import patch
import experiment as common

with tempfile.TemporaryDirectory() as td:
    td=Path(td);state={k:np.zeros_like(v) for k,v in common.initial('mnist',0).items()};state['fc2.bias'][1]=4
    compiler=Compiler('mnist',state,td/'checkpoints')
    class TestAdapter(Adapter):
        def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
            self.proposals+=1;h,_=self.compiler.materialize(candidate)
            return {'fc2':json.dumps({'parent':h,'program':{'ops':[
                {'op':'set','tensor':'fc2.bias','value':0},
                {'op':'set','tensor':'fc2.bias','index':[0],'value':4}]}})}
    xx=np.zeros((8,1,28,28),dtype=np.float32);yy=np.zeros(8,dtype=np.int64)
    adapter=TestAdapter('mnist',0,'cpu',td,compiler,{'feedback':xx,'selection':xx},{'feedback':yy,'selection':yy})
    candidate={m:json.dumps({'parent':'base','program':{'ops':[]}}) for m in common.modules('mnist')}
    result=gepa.optimize(seed_candidate=candidate,
        trainset=[{'split':'feedback','index':i} for i in range(8)],
        valset=[{'split':'selection','index':i} for i in range(8)],adapter=adapter,
        module_selector=lambda *args:['fc2'],reflection_minibatch_size=8,skip_perfect_score=False,
        stop_callbacks=[MaxCandidateProposalsStopper(1)],max_metric_calls=100,run_dir=str(td/'gepa'))
    assert result.num_candidates==2 and max(result.val_aggregate_scores)>.9
    rows=[json.loads(l) for l in (td/'evaluations.jsonl').read_text().splitlines()]
    assert all(not r['capture_traces'] for r in rows if r['split']=='selection')
    # A joint checkpoint is assembled exactly, then a local edit preserves its other layer.
    joint={k:np.full_like(v,2) for k,v in state.items()}
    joint_id='f'*64;np.savez(compiler.folder/f'{joint_id}.npz',**joint)
    joint_candidate={m:json.dumps({'parent':joint_id,'program':{'ops':[]}}) for m in common.modules('mnist')}
    _,assembled=compiler.materialize(joint_candidate)
    assert all(np.array_equal(assembled[k],v) for k,v in joint.items())
    joint_candidate['fc2']=json.dumps({'parent':joint_id,'program':{'ops':[{'op':'set','tensor':'fc2.bias','value':3}]}})
    _,assembled=compiler.materialize(joint_candidate)
    assert np.all(assembled['fc1.weight']==2) and np.all(assembled['fc2.bias']==3)
    partial,status=execute(state,{'ops':[{'op':'fit','tensor':'fc1.weight'},
        {'op':'set','tensor':'fc2.bias','value':2}]},list(state),return_status=True)
    assert status['applied']==1 and len(status['rejected'])==1 and np.all(partial['fc2.bias']==2)
    common.dump(td/'construction.json',{'ops':[]})
    common.dump(td/'construction_status.json',{'error':None})
    adapter=Adapter('mnist',0,'cpu',td,compiler,{'feedback':xx,'selection':xx},{'feedback':yy,'selection':yy})
    program={'hypothesis':'integration check','ops':[
        {'op':'set','tensor':'fc1.weight','value':.25},
        {'op':'set','tensor':'fc2.bias','value':.75}]}
    with patch.object(strict,'call',return_value=program) as request:
        updates=adapter.propose_new_texts(candidate,{'fc1':[],'fc2':[]},['fc1','fc2'])
    assert request.call_count==1 and set(updates)=={'fc1','fc2'}
    _,assembled=compiler.materialize(updates)
    assert np.all(assembled['fc1.weight']==.25) and np.all(assembled['fc2.bias']==.75)
    for program in [
        {'ops':[{'op':'fit','tensor':'fc1.weight'}]},
        {'ops':[{'op':'set','tensor':'fc1.weight','expr':'np.linalg.solve(i0,i1)'}]},
        {'ops':[{'op':'copy','tensor':'fc1.weight','source_tensor':'/tmp/gepa-weight-space-fireworks.key'}]},
    ]:
        try:execute(state,program,list(state))
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Forbidden action accepted')
print('PASS: real GEPA acceptance/full-selection with declarative weights; no selection traces; fitting, Python and file access rejected')
