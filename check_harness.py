"""Meaningful checks of the information boundary and checkpoint compilation."""
import json, tempfile
from pathlib import Path
import numpy as np
from experiment import ROOT, Adapter, Compiler, Network, initial, sandbox, modules, IDENTITY
import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

def main():
    state=initial('mnist',0)
    out=sandbox(state,IDENTITY,list(state))
    assert all(np.array_equal(out[k],v) for k,v in state.items())
    out=sandbox(state,'def edit(w):\n    w["fc1.weight"][:] = np.clip(np.nan_to_num(w["fc1.weight"]), -1, 1)\n    return w\n',list(state))
    assert all(np.isfinite(v).all() for v in out.values())
    out=sandbox(state,'def edit(w):\n    a,b=divmod(11,3)\n    w["fc1.bias"][0]=pow(a,b)\n    return w\n',list(state))
    assert out['fc1.bias'][0]==9
    failures=[
        'def edit(w):\n    w["fc1.bias"][:]=np.nan\n    return w',
        'def edit(w):\n    w["fc2.bias"][:]=1\n    return w',
        'def edit(w):\n    open("/tmp/gepa-weight-space-fireworks.key").read()\n    return w',
        'import os\ndef edit(w):\n    return w',
        'def edit(w):\n    np.load("/home/devichand/gepa-weight-space/data/final_test/mnist.npz")\n    return w',
    ]
    for code in failures:
        try:sandbox(state,code,['fc1.weight','fc1.bias'])
        except ValueError:pass
        else:raise AssertionError('An invalid edit passed')
    with tempfile.TemporaryDirectory() as td:
        c=Compiler('mnist',state,Path(td)/'checkpoints')
        seed={m:json.dumps({'parent':'base','code':IDENTITY}) for m in modules('mnist')}
        h,material=c.materialize(seed)
        child=dict(seed);child['fc1']=json.dumps({'parent':h,'code':'def edit(w):\n    w["fc1.bias"][:]=2\n    return w\n'})
        ch,material=c.materialize(child)
        assert np.all(material['fc1.bias']==2)
        assert np.array_equal(material['fc2.weight'],state['fc2.weight'])
        ch2,again=c.materialize(child);assert ch==ch2
        assert all(np.array_equal(material[k],again[k]) for k in material)
        xx=np.zeros((2,1,28,28),dtype='float32');yy=np.zeros(2,dtype='int64')
        adapter=Adapter('mnist',0,'cpu',Path(td),c,{'feedback':xx,'selection':xx},{'feedback':yy,'selection':yy})
        try:adapter.evaluate([{'split':'selection','index':0}],seed,capture_traces=True)
        except AssertionError:pass
        else:raise AssertionError('Selection content reached reflection')
    assert sum(v.size for v in state.values())==101770
    assert sum(v.size for v in initial('fashion',0).values())==20490
    # Exercise actual GEPA acceptance + full selection evaluation, which uses
    # a different engine path from initial seed evaluation.
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);state={k:np.zeros_like(v) for k,v in initial('mnist',0).items()}
        state['fc2.bias'][1]=4
        compiler=Compiler('mnist',state,td/'checkpoints')
        class IntegrationAdapter(Adapter):
            def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
                self.proposals+=1
                assert all(t.get('split')!='selection' for records in reflective_dataset.values() for t in records)
                h,_=self.compiler.materialize(candidate)
                return {'fc2':json.dumps({'parent':h,'code':'def edit(w):\n    w["fc2.bias"][:]=0\n    w["fc2.bias"][0]=4\n    return w\n'})}
        xx=np.zeros((8,1,28,28),dtype='float32');yy=np.zeros(8,dtype='int64')
        a=IntegrationAdapter('mnist',0,'cpu',td,compiler,{'feedback':xx,'selection':xx},{'feedback':yy,'selection':yy})
        candidate={m:json.dumps({'parent':'base','code':IDENTITY}) for m in modules('mnist')}
        result=gepa.optimize(seed_candidate=candidate,
            trainset=[{'split':'feedback','index':i} for i in range(8)],
            valset=[{'split':'selection','index':i} for i in range(8)],adapter=a,
            module_selector=lambda *args:['fc2'],reflection_minibatch_size=8,skip_perfect_score=False,
            stop_callbacks=[MaxCandidateProposalsStopper(1)],max_metric_calls=100,run_dir=str(td/'gepa'))
        assert result.num_candidates==2 and max(result.val_aggregate_scores)>.9
        rows=[json.loads(line) for line in (td/'evaluations.jsonl').read_text().splitlines()]
        assert all(not r['capture_traces'] for r in rows if r['split']=='selection')
    print('PASS: tensor constraints, no file/import access, immutable reconstruction, selection trace boundary, model sizes')
    print('PASS: real GEPA accepted-candidate full-selection integration, with no selection traces')

if __name__=='__main__':main()
