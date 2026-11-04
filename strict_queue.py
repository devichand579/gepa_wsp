"""Primary strict run queue, three independent seeds per assigned GPU."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def run(task,seed,device):
    frozen=json.loads((ROOT/'runtime/PROTOCOL_FROZEN.json').read_text())
    for name,expected in frozen['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,f'Frozen source changed: {name}'
    folder=ROOT/'runs'/'main'/task/f'seed{seed}';folder.mkdir(exist_ok=True,parents=True)
    result=folder/'search_result.json'
    if result.exists() and json.loads(result.read_text())['proposals']>=40:return
    if result.exists():result.rename(folder/f'search_result_partial_{int(time.time())}.json')
    stop=folder/'gepa/gepa.stop'
    if stop.exists():stop.unlink()
    if not (folder/'adam_result.json').exists():
        with (ROOT/'runtime'/f'main_{task}_seed{seed}_adam.log').open('a') as f:
            subprocess.run([sys.executable,str(ROOT/'experiment.py'),'--task',task,'--seed',str(seed),
                            '--device',device,'--phase','main','--method','adam','--epochs','50'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'runtime'/f'strict_main_{task}_seed{seed}.log').open('a') as f:
        subprocess.run([sys.executable,str(ROOT/'strict_experiment.py'),'--task',task,'--seed',str(seed),
                        '--device',device,'--phase','main','--proposals','40'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert result.exists() and json.loads(result.read_text())['proposals']>=40

def main(task,device):
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('STRICT_CONCURRENCY','3'))) as pool:
        futures=[pool.submit(run,task,seed,device) for seed in [0,1,2]]
        for future in concurrent.futures.as_completed(futures):future.result()
    (ROOT/'runtime'/f'{task}_queue_complete.json').write_text(json.dumps({'task':task,'time':time.time(),'interface':'strict declarative edits'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['mnist','fashion']);p.add_argument('--device',required=True)
    a=p.parse_args();main(a.task,a.device)
