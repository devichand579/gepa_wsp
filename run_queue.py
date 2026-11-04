"""One persistent task queue per GPU. Launch only after pilot/protocol review."""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from experiment import ROOT,dump

def queue(task,device):
    assert (ROOT/'runtime'/'PROTOCOL_FROZEN.json').exists(),'Freeze the protocol before main runs'
    assert not (ROOT/'runs'/'main'/'SELECTION_FROZEN.json').exists(),'Final evaluation already started'
    frozen=json.loads((ROOT/'runtime'/'PROTOCOL_FROZEN.json').read_text())
    for seed in [0,1,2]:
        for method in ['adam','search']:
            for name,expected in frozen['source_hashes'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,f'Frozen source changed: {name}'
            ledger=json.loads((ROOT/'runtime'/'usage.json').read_text())
            if ledger['blocked']:raise RuntimeError('Service/billing block: no more experiment dispatch')
            target=ROOT/'runs'/'main'/task/f'seed{seed}'/('adam_result.json' if method=='adam' else 'search_result.json')
            if target.exists():continue
            log=ROOT/'runtime'/f'main_{task}_seed{seed}_{method}.log'
            cmd=[sys.executable,str(ROOT/'experiment.py'),'--task',task,'--seed',str(seed),'--device',device,
                 '--phase','main','--method',method,'--proposals','40','--epochs','50']
            with log.open('a') as f:
                subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            assert target.exists(),f'Missing run output: {target}'
        dump(ROOT/'runtime'/f'{task}_queue_status.json',{'task':task,'last_completed_seed':seed,'time':time.time()})
    dump(ROOT/'runtime'/f'{task}_queue_complete.json',{'task':task,'time':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['mnist','fashion'],required=True);p.add_argument('--device',required=True)
    a=p.parse_args();queue(a.task,a.device)
