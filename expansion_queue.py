"""Local follow-up scheduling; no continuous Codex agent polling is needed."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import experiment as common
ROOT=common.ROOT
TASKS=['mnist','fashion','sst2']

def live(name):return subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0

def _base_jobs(group):
 if group=='classification':
  for task in TASKS:
   for seed in [0,1,2]:
    yield {'id':f'deepseek60_{task}_{seed}','folder':ROOT/'expansion_runs/classification/deepseek60'/task/f'seed{seed}','budget':60,'prefixes':[0,5,10,30,60],'args':['classification_followup.py','--kind','continue','--model','deepseek','--task',task,'--seed',str(seed),'--device','cuda:0']}
  for model in ['nemotron','minimax','deepseek']:
   for task in TASKS:
    for seed in [0,1,2]:
     yield {'id':f'{model}_{task}_{seed}','folder':ROOT/'expansion_runs/classification'/('deepseek10' if model=='deepseek' else model)/task/f'seed{seed}','budget':10,'prefixes':[0,5,10],'args':['classification_followup.py','--kind','models','--model',model,'--task',task,'--seed',str(seed),'--device','cuda:0']}
 else:
  for model in ['nemotron','minimax','deepseek']:
   for seed in [0,1,2]:
    yield {'id':f'{model}_diffusion_{seed}','folder':ROOT/'diffusion_runs/models'/model/f'seed{seed}','budget':20,'prefixes':[0,5,10,20],'args':['diffusion_experiment.py','--model',model,'--seed',str(seed),'--device','cuda:1']}

def jobs(group):
 path=ROOT/'runtime/EXPERIMENT_REPAIRS.json';repairs=json.loads(path.read_text()) if path.exists() else {}
 for job in _base_jobs(group):
  if job['id'] in repairs:
   repair=repairs[job['id']];job['folder']=ROOT/repair['folder']
   job['args']=['repair_experiment.py','--job',job['id'],'--model',repair['model'],'--device','cuda:1' if group=='diffusion' else 'cuda:0']
   if group=='classification':job['args']+=['--kind','continue']
  yield job

def retire_previous_dispatcher(group):
 with (ROOT/'runtime'/f'retire_{group}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  _retire_previous_dispatcher(group)

def _retire_previous_dispatcher(group):
 path=ROOT/'runtime'/f'PRE_REPAIR_{group.upper()}_CONTROLLERS.json'
 marker=ROOT/'runtime'/f'PRE_REPAIR_{group.upper()}_CONTROLLERS_RETIRED.json'
 if not path.exists() or marker.exists():return
 records=json.loads(path.read_text())
 for record in records:
  pid=record['pid']
  while True:
   active=False
   for child in record['children']:
    p=Path('/proc')/child
    try:
     state=(p/'status').read_text()
     if '\nState:\tZ' not in state:active=True
    except FileNotFoundError:pass
   if not active:break
   time.sleep(30)
  p=Path('/proc')/str(pid)
  try:
   args=[a.decode() for a in (p/'cmdline').read_bytes().split(b'\0') if a]
   if args==record['cmdline'] and (p/'cwd').resolve()==ROOT:os.kill(pid,signal.SIGKILL)
  except (FileNotFoundError,ProcessLookupError):pass
 common.dump(marker,{'time':time.time()})

def verify_job(job):
 result=job['folder']/'search_result.json'
 if not result.exists():return False
 data=json.loads(result.read_text())
 if data['proposals']!=job['budget']:return False
 for n in job['prefixes']:
  path=job['folder']/'prefixes'/f'budget{n:03d}.json';weights=path.with_suffix('.npz')
  if not path.exists() or not weights.exists():return False
  assert hashlib.sha256(weights.read_bytes()).hexdigest()==json.loads(path.read_text())['weights_sha256']
 return True

def job_lock_path(job):
 folder=ROOT/'runtime/job_locks';folder.mkdir(exist_ok=True)
 return folder/(job['id']+'.lock')

def job_finished(job):
 # Avoid reading prefix files while another worker is still materializing them.
 with job_lock_path(job).open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  return verify_job(job)

def run_job(job):
 with job_lock_path(job).open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return False
  _run_job(job)
  return True

def _run_job(job):
 attempts=0
 while not verify_job(job):
  ledger=json.loads((ROOT/'runtime/usage.json').read_text())
  if ledger['blocked']:
   if any(c['status']=='unknown_billing' for c in ledger['calls']) and live('gepa_rich_monitor'):time.sleep(60);continue
   raise RuntimeError('Global budget/service block; no automatic cap override')
  folder=job['folder'];marker=folder/'llm/provider_circuit.json'
  if (folder/'llm/provider_nonretryable.json').exists():raise RuntimeError('Provider configuration rejected: '+job['id'])
  if marker.exists():
   until=json.loads(marker.read_text())['retry_after']
   if time.time()<until:time.sleep(min(60,until-time.time()));continue
  partial=folder/'search_result.json'
  if partial.exists():partial.rename(folder/f'search_result_partial_{int(time.time())}.json')
  stop=folder/'gepa/gepa.stop'
  if stop.exists():stop.unlink()
  freeze=json.loads((ROOT/'runtime/EXPANSION_FROZEN.json').read_text())
  for name,sha in freeze['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
  common.dump(ROOT/'runtime'/f'expansion_{job["id"]}_active.json',{'time':time.time(),'folder':str(folder),'attempt':attempts})
  with (ROOT/'runtime'/f'expansion_{job["id"]}.log').open('a') as log:
   result=subprocess.run([sys.executable,*[str(ROOT/a) if i==0 else a for i,a in enumerate(job['args'])]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  if verify_job(job):break
  attempts+=1
  if attempts>3:raise RuntimeError('Bounded recovery attempts exhausted: '+job['id'])
  if marker.exists():continue
  ledger=json.loads((ROOT/'runtime/usage.json').read_text())
  if ledger['blocked']:continue
  raise RuntimeError(f'Follow-up worker exited({result.returncode}) before required outputs: '+job['id'])

def freeze_selections():
 with (ROOT/'runtime/expansion_freeze.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (ROOT/'runtime/EXPANSION_SELECTIONS_FROZEN.json').exists():return
  return _freeze_selections_locked()

def _freeze_selections_locked():
 for group in ['classification','diffusion']:
  if not (ROOT/'runtime'/f'EXPANSION_{group.upper()}_COMPLETE.json').exists():return
 records={}
 for group in ['classification','diffusion']:
  for job in jobs(group):
   assert verify_job(job),job['id']
   records[job['id']]={str(n):json.loads((job['folder']/'prefixes'/f'budget{n:03d}.json').read_text()) for n in job['prefixes']}
 plan=json.loads((ROOT/'runtime/EXPANSION_PLAN.json').read_text())
 assert hashlib.sha256((ROOT/plan['plan']).read_bytes()).hexdigest()==plan['plan_sha256']
 common.dump(ROOT/'runtime/EXPANSION_SELECTIONS_FROZEN.json',{'time':time.time(),'plan_sha256':plan['plan_sha256'],'selections':records})
 # Only the second queue to finish reaches here. The finalizer uses its own lock.
 finalizer=ROOT/'expansion_finalize.py'
 if finalizer.exists():subprocess.run([sys.executable,str(finalizer)],cwd=ROOT,check=True)
 else:common.dump(ROOT/'runtime/EXPANSION_REPORT_PENDING.json',{'time':time.time(),'reason':'All expanded selections frozen; combined report finalizer still needs implementation.'})

def main(group,device=None):
 retire_previous_dispatcher(group)
 if group=='classification':
  while not all((ROOT/'runtime'/f'rich_{task}_complete.json').exists() for task in TASKS):
   if not live('gepa_rich_monitor'):raise RuntimeError('Primary classification monitor is missing before completion')
   time.sleep(60)
 for slug in ['minimax','nemotron']:assert json.loads((ROOT/'expansion_runs/probes'/slug/'result.json').read_text())['passed']
 while True:
  planned=list(jobs(group))
  if all(job_finished(job) for job in planned):break
  for job in planned:
   if device is not None:job['args'][job['args'].index('--device')+1]=device
   run_job(job)
  if not all(job_finished(job) for job in planned):time.sleep(30)
 with (ROOT/'runtime/expansion_completion.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  common.dump(ROOT/'runtime'/f'EXPANSION_{group.upper()}_COMPLETE.json',{'time':time.time()})
 freeze_selections()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',choices=['classification','diffusion'],required=True);p.add_argument('--device');p.add_argument('--help-classification',action='store_true');a=p.parse_args();active_group=a.group
 try:
  main(active_group,a.device)
  if active_group=='diffusion' and a.help_classification:
   active_group='classification';main(active_group,a.device or 'cuda:1')
 except Exception as e:common.dump(ROOT/'runtime'/f'EXPANSION_{active_group.upper()}_ATTENTION.json',{'time':time.time(),'reason':str(e)});raise
