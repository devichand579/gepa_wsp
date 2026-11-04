"""Run the rich suite, bounded provider recovery, audit and final report."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
import experiment as common
ROOT=common.ROOT
TASKS={'mnist':'cuda:0','fashion':'cuda:1','sst2':'cuda:0'}

def alive(name):return subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0
def launch():
 for task,device in TASKS.items():
  if (ROOT/'runtime'/f'rich_{task}_complete.json').exists() or alive('gepa_rich_'+task):continue
  command='REFLECTION_MODEL=accounts/fireworks/models/deepseek-v4p1-flash .venv/bin/python -u rich_queue.py --task '+task+' --device '+device+' >> runtime/rich_'+task+'_queue.log 2>&1'
  subprocess.run(['tmux','new-session','-d','-s','gepa_rich_'+task,'-c',str(ROOT),command],check=True)

def settle_unknown():
 r=ROOT/'runtime'
 with (r/'usage.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads((r/'usage.json').read_text())
  assert not any(c['status']=='pending' for c in ledger['calls'])
  unknown=[(i,c) for i,c in enumerate(ledger['calls']) if c['status']=='unknown_billing'];assert unknown
  common.dump(r/f'rich_ledger_before_settlement_{int(time.time())}.json',ledger)
  lookup={json.loads(p.read_text())['ledger_call']:p.parent for p in (ROOT/'rich_runs').rglob('dispatch.json')}
  settled=[]
  for i,c in unknown:
   assert i in lookup and c['reserved']>0 and c['label'].startswith('rich/')
   bound=c['reserved'];c.update(status='conservative_settlement',cost=bound,actual_billing_unknown=True,settled_at=time.time())
   ledger['spent']+=bound;ledger['reserved']-=bound
   common.dump(lookup[i]/'service_failure.json',{'ledger_call':i,'error_type':c.get('error'),'actual_billing_unknown':True,'conservative_charge':bound})
   settled.append({'ledger_call':i,'upper_bound_charged':bound})
  if abs(ledger['reserved'])<1e-9:ledger['reserved']=0.
  assert ledger['spent']+ledger['reserved']<=ledger['cap'] and len(ledger['calls'])<ledger['max_calls']
  ledger['blocked']=False;ledger.pop('blocked_reason',None);common.dump(r/'usage.json',ledger)
  common.append(r/'service_settlements.jsonl',{'protocol':'rich','time':time.time(),'changes':settled,'conservative_total':ledger['spent']})

def status():
 ledger=json.loads((ROOT/'runtime/usage.json').read_text());out={'active_protocol':'feedback-aware NumPy weight programs','time':time.time(),'calls':len(ledger['calls']),'conservatively_accounted_cost':ledger['spent'],'reserved':ledger['reserved'],'blocked':ledger['blocked'],'runs':[]}
 for task in TASKS:
  for seed in [0,1,2]:
   p=ROOT/'rich_runs/rich_main'/task/f'seed{seed}';row={'task':task,'seed':seed,'dispatched_mutations':common.mutation_api_calls(p)}
   for name in ['one_shot_selection.json','search_result.json']:
    if (p/name).exists():row[name]=json.loads((p/name).read_text())
   out['runs'].append(row)
 common.dump(ROOT/'runtime/RICH_STATUS.json',out);common.dump(ROOT/'runtime/STATUS.json',out);return ledger

def main():
 recoveries=0;launch()
 while True:
  ledger=status()
  if all((ROOT/'runtime'/f'rich_{task}_complete.json').exists() for task in TASKS):break
  if ledger['blocked']:
   if any(c['status']=='pending' for c in ledger['calls']) or any(alive('gepa_rich_'+t) for t in TASKS):time.sleep(30);continue
   if recoveries>=3 or not any(c['status']=='unknown_billing' for c in ledger['calls']):raise RuntimeError('Budget/service block requires attention; no further automatic override')
   settle_unknown();recoveries+=1;launch();continue
  for task in TASKS:
   if not (ROOT/'runtime'/f'rich_{task}_complete.json').exists() and not alive('gepa_rich_'+task):raise RuntimeError(task+' queue exited before completion')
  time.sleep(30)
 subprocess.run([sys.executable,str(ROOT/'rich_finalize.py')],cwd=ROOT,check=True)
 ready={'time':time.time(),'active_protocol':'rich-feedback','blog':str(ROOT/'BLOG_RICH.md'),'results':str(ROOT/'rich_runs/rich_main/FINAL_RESULTS.json')}
 common.dump(ROOT/'runtime/RICH_RESULTS_READY.json',ready);common.dump(ROOT/'runtime/ALL_RESULTS_READY.json',ready)
 print('RICH_RESULTS_READY',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  common.dump(ROOT/'runtime/RICH_ATTENTION.json',{'time':time.time(),'reason':str(e)});raise
