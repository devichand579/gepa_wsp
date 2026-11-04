"""Conservative accounting and bounded restart after provider failures.

No unknown usage is treated as zero: charge its entire pre-dispatch reservation.
Failed requests remain failed proposals and are never retried with the same ID.
"""
import argparse,fcntl,json,math,shlex,subprocess,time
from pathlib import Path
import experiment as common
ROOT=common.ROOT

def settle():
 r=ROOT/'runtime'
 with (r/'usage.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);d=json.loads((r/'usage.json').read_text())
  assert not any(c['status']=='pending' for c in d['calls']),'Wait for in-flight requests'
  unknown=[(i,c) for i,c in enumerate(d['calls']) if c['status']=='unknown_billing']
  assert unknown,'No unknown billing to settle; do not unblock other failures'
  common.dump(r/f'ledger_before_settlement_{int(time.time())}.json',d)
  changes=[]
  for i,c in unknown:
   bound=c['reserved'];assert math.isfinite(bound) and bound>0 and 'prices' in c
   _,task,seed,mutation=c['label'].split('/')
   folder=(ROOT/'text_runs/sst2' if task=='sst2' else ROOT/'runs/main'/task)/seed/'llm'/f'mutation_{int(mutation.removeprefix("mutation")):03d}'
   # Current failures are primary-suite dispatches; preserve the ID at its actual request.
   candidates=[p for p in folder.rglob('dispatch.json') if json.loads(p.read_text()).get('ledger_call')==i]
   if candidates:folder=candidates[0].parent
   assert (folder/'request.json').exists(),str(folder)
   common.dump(folder/'dispatch.json',{'ledger_call':i,'label':c['label'],'time':c['time'],'backfilled_after_service_failure':True})
   common.dump(folder/'service_failure.json',{'ledger_call':i,'error_type':c.get('error'),'actual_billing_unknown':True,'conservative_charge':bound})
   c.update(status='conservative_settlement',cost=bound,actual_billing_unknown=True,settled_at=time.time())
   d['spent']+=bound;d['reserved']-=bound;changes.append({'ledger_call':i,'charged_upper_bound':bound})
  if abs(d['reserved'])<1e-9:d['reserved']=0.
  assert d['spent']+d['reserved']<=d['cap'] and len(d['calls'])<d['max_calls']
  d['blocked']=False;d.pop('blocked_reason',None);common.dump(r/'usage.json',d)
  common.append(r/'service_settlements.jsonl',{'time':time.time(),'changes':changes,'conservative_total':d['spent']})
 print(json.dumps({'settled':changes,'conservative_total':d['spent']}),flush=True)

def launch():
 r=ROOT/'runtime';p=r/'ATTENTION.json'
 if p.exists():p.rename(r/f'ATTENTION_service_{int(time.time())}.json')
 env='REFLECTION_MODEL=accounts/fireworks/models/deepseek-v4p1-flash REFLECTION_EFFORT=low STRICT_THINKING_BUDGET=8192 STRICT_MAX_OUTPUT=65536 STRICT_CONCURRENCY=1 '
 jobs={'gepa_mnist':env+'.venv/bin/python -u strict_queue.py --task mnist --device cuda:0 >> runtime/strict_mnist_queue.log 2>&1',
       'gepa_fashion':env+'.venv/bin/python -u strict_queue.py --task fashion --device cuda:1 >> runtime/strict_fashion_queue.log 2>&1',
       'gepa_text':env+'.venv/bin/python -u text_queue.py >> runtime/text_queue.log 2>&1',
       'gepa_supervisor':'.venv/bin/python -u supervise.py >> runtime/strict_supervisor.log 2>&1',
       'gepa_report':'.venv/bin/python -u assemble_blog.py >> runtime/assemble_blog.log 2>&1'}
 for name,command in jobs.items():
  if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode:subprocess.run(['tmux','new-session','-d','-s',name,'-c',str(ROOT),command],check=True)
 print('Launched checkpoint resumes with at most three concurrent provider requests',flush=True)

def watch():
 for attempt in range(4):
  while True:
   d=json.loads((ROOT/'runtime/usage.json').read_text())
   if (ROOT/'runtime/ALL_RESULTS_READY.json').exists():return
   if d['blocked']:
    if any(c['status']=='pending' for c in d['calls']):time.sleep(30);continue
    if not any(c['status']=='unknown_billing' for c in d['calls']):raise RuntimeError('Non-provider budget block; no automatic override')
    # Let all old controllers finish before resuming their checkpoints.
    alive=any(subprocess.run(['tmux','has-session','-t',n],capture_output=True).returncode==0 for n in ['gepa_mnist','gepa_fashion','gepa_text'])
    if alive:time.sleep(30);continue
    if attempt==3:raise RuntimeError('Three additional provider recoveries exhausted')
    settle();launch();break
   time.sleep(30)
 raise RuntimeError('Three additional provider recoveries exhausted; inspect runtime logs')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--settle',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
 if a.settle:settle()
 if a.launch:launch()
 if a.watch:watch()
