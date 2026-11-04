"""Restart-safe independent seeds on one GPU, using the frozen learning protocol."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def run(task,seed,device):
    frozen=json.loads((ROOT/'runtime/PROTOCOL_FROZEN.json').read_text())
    for name,expected in frozen['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,f'Frozen source changed: {name}'
    folder=ROOT/'runs'/'main'/task/f'seed{seed}'
    result=folder/'search_result.json'
    if result.exists():
        if json.loads(result.read_text())['proposals']>=40:return
        result.rename(folder/f'search_result_partial_{int(time.time())}.json')
    stop=folder/'gepa'/'gepa.stop'
    if stop.exists():stop.unlink()
    for method in ['adam','search']:
        ledger=json.loads((ROOT/'runtime/usage.json').read_text())
        if ledger['blocked']:raise RuntimeError('Service/billing stop')
        target=folder/('adam_result.json' if method=='adam' else 'search_result.json')
        if target.exists():continue
        cmd=[sys.executable,str(ROOT/'experiment.py'),'--task',task,'--seed',str(seed),'--device',device,
             '--phase','main','--method',method,'--proposals','40','--epochs','50']
        with (ROOT/'runtime'/f'main_{task}_seed{seed}_{method}.log').open('a') as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        assert target.exists(),f'Missing result: {target}'
    assert json.loads(result.read_text())['proposals']>=40,'Partial run must not trigger final evaluation'

def main(task,device):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,task,seed,device) for seed in [0,1,2]]
        for future in concurrent.futures.as_completed(futures):future.result()
    (ROOT/'runtime'/f'{task}_queue_complete.json').write_text(json.dumps({'task':task,'time':time.time(),'concurrent_seeds':3}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['mnist','fashion']);p.add_argument('--device',required=True)
    a=p.parse_args();main(a.task,a.device)
