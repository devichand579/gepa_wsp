"""Budget prefixes and independent model/continuation runs; no final-test access."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import torch
import experiment as common
import rich_experiment as rich
import expansion_client as client
ROOT=common.ROOT

def dispatches(folder):
 unique={}
 for p in (folder/'llm').glob('mutation_*/dispatch.json'):
  d=json.loads(p.read_text());unique[d['ledger_call']]=d
 return sorted(unique.values(),key=lambda d:d['ledger_call'])

def select_prefix(folder,budget):
 calls=dispatches(folder)
 assert len(calls)>=budget,'Requested prefix has not run'
 if len(calls)==budget:
  assert (folder/'search_result.json').exists(),'Last in-flight dispatch is not a completed prefix'
  cutoff=float('inf')
 else:cutoff=calls[budget]['time']
 rows=[]
 for line in (folder/'evaluations.jsonl').read_text().splitlines():
  r=json.loads(line)
  if r['split']=='selection' and not r.get('error') and r['time']<cutoff:rows.append(r)
 assert rows,'No valid selection checkpoint before cutoff'
 best=max(rows,key=lambda r:r['metrics']['score'])
 return {'budget':budget,'candidate':best['candidate'],'selection':best['metrics'],'cutoff_before_next_dispatch':None if cutoff==float('inf') else cutoff,'completed_selection_evaluations':len(rows)}

def save_prefixes(folder,budgets):
 out=folder/'prefixes';out.mkdir(exist_ok=True)
 for budget in budgets:
  record=select_prefix(folder,budget);source=folder/'checkpoints'/f'{record["candidate"]}.npz';assert source.exists()
  dest=out/f'budget{budget:03d}.npz'
  if not dest.exists():shutil.copy2(source,dest)
  record['weights_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest();common.dump(out/f'budget{budget:03d}.json',record)

def run(kind,slug,task,seed,device):
 source=ROOT/'rich_runs/rich_main'/task/f'seed{seed}'
 if kind=='continue':
  assert slug=='deepseek';budget=60;folder=ROOT/'expansion_runs/classification/deepseek60'/task/f'seed{seed}'
  result=json.loads((source/'search_result.json').read_text());assert result['proposals']==30
  save_prefixes(source,[0,5,10,30])
  if not folder.exists():
   folder.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,folder)
   (folder/'search_result.json').rename(folder/'search_result_budget30.json')
   if (folder/'run_config.json').exists():(folder/'run_config.json').rename(folder/'run_config_budget30.json')
   # Metadata points to preserved numeric parents; never hard-link mutable logs.
   common.dump(folder/'continuation_provenance.json',{'source':str(source.relative_to(ROOT)),'copied_at':time.time(),'from_budget':30,'to_budget':60,'original_result_sha256':hashlib.sha256((source/'search_result.json').read_bytes()).hexdigest()})
 else:
  assert slug in client.MODELS;budget=10;folder=ROOT/'expansion_runs/classification'/('deepseek10' if slug=='deepseek' else slug)/task/f'seed{seed}'
  if slug!='deepseek':
   probe=json.loads((ROOT/'expansion_runs/probes'/slug/'result.json').read_text());assert probe['passed'],'Model capability probe must pass first'
  folder.mkdir(parents=True,exist_ok=True)
  for name in ['adam_best.npz','adam_result.json']:
   if not (folder/name).exists():shutil.copy2(source/name,folder/name)
 client.configure(slug,compact=kind=='models')
 stop=folder/'gepa/gepa.stop'
 if stop.exists():stop.unlink()
 common.dump(folder/'expansion_config.json',{'kind':kind,'model':client.MODELS[slug],'task':task,'seed':seed,'budget':budget,'compact_context':kind=='models','source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['classification_followup.py','expansion_client.py','rich_experiment.py','rich_worker.py']}})
 rich.run(task,seed,device,folder,budget)
 assert json.loads((folder/'search_result.json').read_text())['proposals']==budget
 save_prefixes(folder,[0,5,10,30,60] if kind=='continue' else [0,5,10])
 common.dump(folder/'FOLLOWUP_COMPLETE.json',{'time':time.time(),'budget':budget})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['models','continue'],required=True);p.add_argument('--model',choices=list(client.MODELS),required=True);p.add_argument('--task',choices=['mnist','fashion','sst2'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2)
 run(a.kind,a.model,a.task,a.seed,a.device)
