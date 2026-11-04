"""Select nested budget prefixes using development evaluations only."""
import hashlib, json, shutil


import core as common


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
