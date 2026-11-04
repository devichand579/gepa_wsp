"""Complete evaluator/report automatically once both local experiment queues finish."""
import json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    start=time.time()
    while True:
        ledger_path=ROOT/'runtime/usage.json'
        ledger=json.loads(ledger_path.read_text())
        status={'time':time.time(),'elapsed_seconds':time.time()-start,
                'api_calls':len(ledger['calls']),'estimated_api_cost':ledger['spent'],
                'billing_blocked':ledger['blocked'],'runs':[]}
        for task in ['mnist','fashion']:
            for seed in [0,1,2]:
                folder=ROOT/'runs'/'main'/task/f'seed{seed}'
                record={'task':task,'seed':seed}
                for file,key in [('one_shot_selection.json','one_shot'),('search_result.json','gepa'),('adam_result.json','adam')]:
                    p=folder/file
                    if p.exists():
                        d=json.loads(p.read_text());record[key]=d
                record['proposals_submitted']=len(list((folder/'llm').glob('mutation_*/request.json')))
                status['runs'].append(record)
        tmp=ROOT/'runtime/STATUS.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(ROOT/'runtime/STATUS.json')
        if ledger['blocked']:
            (ROOT/'runtime/ATTENTION.json').write_text(json.dumps({'reason':'Service or budget block','status':status},indent=2))
            return 2
        if all((ROOT/'runtime'/f'{task}_queue_complete.json').exists() for task in ['mnist','fashion']):
            break
        # A crashed queue must be visible, not silently treated as an ongoing run.
        for task in ['mnist','fashion']:
            complete=(ROOT/'runtime'/f'{task}_queue_complete.json').exists()
            alive=subprocess.run(['tmux','has-session','-t',f'gepa_{task}'],capture_output=True).returncode==0
            if not alive and not complete:
                (ROOT/'runtime/ATTENTION.json').write_text(json.dumps({'reason':f'{task} queue exited before completion','status':status},indent=2))
                return 2
        time.sleep(30)
    for script,args in [('audit.py',[]),('finalize.py',['--device','cuda:0']),('report.py',[]),('audit.py',[])]:
        subprocess.run([sys.executable,str(ROOT/script)]+args,cwd=ROOT,check=True)
    (ROOT/'runtime/RESULTS_READY.json').write_text(json.dumps({'time':time.time(),'results':'runs/main/FINAL_RESULTS.json','blog':'BLOG.md'},indent=2))
    print('RESULTS_READY: final held-out results, figures and draft blog produced.',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
