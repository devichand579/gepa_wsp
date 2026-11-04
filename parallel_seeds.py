"""Run independent remaining seeds concurrently without altering the frozen method.

Pause only the two dispatch parents, leaving their existing seed-0 children
running. Resume those parents after seeds 1/2 finish so their normal completion
markers and supervisor behavior remain intact. No duplicate seed dispatch.
"""
import concurrent.futures,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def run(task,seed,device):
    for method in ['adam','search']:
        ledger=json.loads((ROOT/'runtime/usage.json').read_text())
        if ledger['blocked']:raise RuntimeError('Global service/budget stop')
        frozen=json.loads((ROOT/'runtime/PROTOCOL_FROZEN.json').read_text())
        for name,expected in frozen['source_hashes'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
        target=ROOT/'runs'/'main'/task/f'seed{seed}'/('adam_result.json' if method=='adam' else 'search_result.json')
        if target.exists():continue
        command=[sys.executable,str(ROOT/'experiment.py'),'--task',task,'--seed',str(seed),'--device',device,
                 '--phase','main','--method',method,'--proposals','40','--epochs','50']
        with (ROOT/'runtime'/f'main_{task}_seed{seed}_{method}.log').open('a') as f:
            subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    return {'task':task,'seed':seed,'completed':time.time()}

def main():
    parents=[]
    for task in ['mnist','fashion']:
        matches=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:args=(proc/'cmdline').read_bytes().split(b'\0')
            except (OSError,PermissionError):continue
            if str(ROOT/'run_queue.py').encode() in args and task.encode() in args:
                matches.append(int(proc.name))
        assert len(matches)==1,(task,matches)
        parents.append(matches[0])
    record={'time':time.time(),'paused_dispatch_parents':parents,
            'note':'Independent seeds concurrent within each assigned GPU; prompts, budgets, splits, and frozen training code unchanged.'}
    (ROOT/'runtime/parallel_execution.json').write_text(json.dumps(record,indent=2))
    for pid in parents:os.kill(pid,signal.SIGSTOP)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run,task,seed,device) for task,device in [('mnist','cuda:0'),('fashion','cuda:1')] for seed in [1,2]]
            results=[f.result() for f in concurrent.futures.as_completed(futures)]
        (ROOT/'runtime/parallel_complete.json').write_text(json.dumps(results,indent=2))
    finally:
        for pid in parents:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass

if __name__=='__main__':main()
