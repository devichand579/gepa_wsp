"""Process-local multi-model client sharing the inclusive project cost ledger."""
from __future__ import annotations
import json,os,time,urllib.request,fcntl
from pathlib import Path
import numpy as np
import core as common
import classification as rich
MODELS={
 'deepseek':{'id':'accounts/fireworks/models/deepseek-v4p1-flash','prices':[.22,.66]},
 'minimax':{'id':'accounts/fireworks/models/minimax-m3p1-flash','prices':[.15,.60]},
 'nemotron':{'id':'accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b','prices':[.05,.20]},
}
COMPACT_CONTEXT=False

def configure(slug,compact=False):
 global COMPACT_CONTEXT
 COMPACT_CONTEXT=compact
 info=MODELS[slug];common.PRICES[info['id']]=tuple(info['prices']);common.MODEL=info['id']
 # Provider selection affects this worker process.
 rich.call=call
 return info

def compact_prompt(prompt):
 """Same bounded representation for every fresh model comparison; arrays stay exact."""
 start=prompt.index('{"network":');prefix=prompt[:start];payload=json.loads(prompt[start:])
 def rounded(value):
  if isinstance(value,float):return round(value,4)
  if isinstance(value,list):return [rounded(v) for v in value]
  if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
  return value
 payload=rounded(payload);weights=payload.get('feedback',{}).get('current_weights',{})
 for info in weights.values():
  shape=info.get('shape',[]);size=int(np.prod(shape)) if shape else 0
  if size>4096:
   values=info.pop('values_rounded',None)
   if values is not None and len(shape)==2:
    n=min(len(values),max(1,4096//shape[1]));info['selected_rows_rounded']={'indices':list(range(n)),'values':values[:n]}
 payload['context_representation']='Compact display only; edit(w,f) retains every full-precision parameter and permitted feedback array.'
 def formatted():return prefix+json.dumps(payload)
 if len(formatted().encode())>180000:
  for item in payload.get('recent_proposals',[]):
   if len(json.dumps(item.get('program_statistics',{})))>4000:item['program_statistics']={'context_note':'Older numeric report omitted; current arrays remain fully accessible.'}
 if len(formatted().encode())>180000:
  for info in weights.values():
   for name in ['values_rounded','selected_rows_rounded','first_filters','values']:info.pop(name,None)
 assert len(formatted().encode())<=180000,'Compact prompt still exceeds the declared common context budget'
 return formatted()

def settle_failed_dispatch(index,error):
 budget=common.Budget()
 with budget.lock.open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);ledger=json.loads(budget.path.read_text());entry=ledger['calls'][index]
  assert entry['status']=='pending'
  bound=entry['reserved'];entry.update(status='conservative_settlement',cost=bound,actual_billing_unknown=True,error=error,settled_at=time.time())
  ledger['spent']+=bound;ledger['reserved']-=bound
  if abs(ledger['reserved'])<1e-9:ledger['reserved']=0.
  # Preserve any block from another request; never silently clear it.
  common.dump(budget.path,ledger)

def check_provider_circuit(folder):
 llm=folder.parent if folder.parent.name=='llm' else folder
 marker=llm/'provider_circuit.json';reset=llm/'provider_circuit_reset.json'
 if (llm/'provider_nonretryable.json').exists():raise RuntimeError('Provider rejected request configuration; fix before further dispatch')
 if marker.exists():
  info=json.loads(marker.read_text())
  if time.time()<info['retry_after']:raise RuntimeError('Provider circuit open; local queue waits before resuming')
  common.dump(reset,{'after_ledger_call':info['last_ledger_call']});marker.unlink()
 after=json.loads(reset.read_text())['after_ledger_call'] if reset.exists() else -1
 calls=[]
 for path in llm.glob('mutation_*/dispatch.json'):
  entry=json.loads(path.read_text())
  if entry['ledger_call']>after:calls.append((entry['ledger_call'],path.parent))
 calls.sort()
 if len(calls)>=3 and all((path/'service_failure.json').exists() for _,path in calls[-3:]):
  common.dump(marker,{'opened_at':time.time(),'retry_after':time.time()+300,'last_ledger_call':calls[-1][0]})
  stop=llm.parent/'gepa/gepa.stop';stop.parent.mkdir(exist_ok=True);stop.write_text('Provider circuit: resume after cooldown, preserving paid-call accounting.\n')
  raise RuntimeError('Three provider failures; local queue must cool down')

def call(prompt,folder,label,*,system=None,max_tokens=32768,thinking_budget=8192):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 if (folder/'service_failure.json').exists():raise RuntimeError('Recorded provider failure is not retried')
 check_provider_circuit(folder)
 if COMPACT_CONTEXT:prompt=compact_prompt(prompt)
 secret=Path(os.environ.get('FIREWORKS_API_KEY_FILE','/tmp/gepa-weight-space-fireworks.key')).read_text().strip();assert secret not in prompt
 payload={'model':common.MODEL,'messages':[{'role':'system','content':system or rich.SYSTEM},{'role':'user','content':prompt}],
          'temperature':.6,'max_tokens':max_tokens,'thinking':{'type':'enabled','budget_tokens':thinking_budget}}
 if (folder/'response.json').exists():
  assert json.loads((folder/'request.json').read_text())==payload,'Request changed after dispatch'
  common.append(folder/'replays.jsonl',{'time':time.time(),'exact_request_replay':True})
  return common.parse_response(json.loads((folder/'response.json').read_text()))
 common.dump(folder/'request.json',payload)
 budget=common.Budget();index=budget.reserve(label,len(json.dumps(payload['messages']).encode()),max_tokens)
 common.dump(folder/'dispatch.json',{'ledger_call':index,'label':label,'model':common.MODEL,'time':time.time()})
 request=urllib.request.Request('https://api.fireworks.ai/inference/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
 started=time.time()
 try:
  with urllib.request.urlopen(request,timeout=600) as response:body=json.load(response)
 except Exception as error:
  info={'ledger_call':index,'error_type':type(error).__name__,'http_status':getattr(error,'code',None),'charged_full_reservation':True}
  if hasattr(error,'read'):
   try:info['provider_message']=error.read(12000).decode(errors='replace').replace(secret,'[REDACTED]')
   except Exception:pass
  if info['http_status'] in [400,401,403,404,413,422]:
   control=folder.parent if folder.parent.name=='llm' else folder
   common.dump(control/'provider_nonretryable.json',info)
   if control.name=='llm':
    stop=control.parent/'gepa/gepa.stop';stop.parent.mkdir(exist_ok=True);stop.write_text('Provider request rejected; inspect configuration before resuming.\n')
  common.dump(folder/'service_failure.json',info);settle_failed_dispatch(index,type(error).__name__)
  raise RuntimeError('Provider failure; full reservation charged; no retry') from None
 assert secret not in json.dumps(body);usage=body.get('usage')
 if not usage or not all(k in usage for k in ['prompt_tokens','completion_tokens']):
  common.dump(folder/'unbilled_response.json',body);common.dump(folder/'service_failure.json',{'ledger_call':index,'error_type':'missing_usage','charged_full_reservation':True});settle_failed_dispatch(index,'missing_usage');raise RuntimeError('Missing usage; full reservation charged')
 budget.finish(index,usage);common.dump(folder/'response.json',body)
 common.dump(folder/'meta.json',{'ledger_call':index,'model':common.MODEL,'elapsed':time.time()-started,'usage':usage,'finish_reason':body['choices'][0].get('finish_reason')})
 (folder/'answer.md').write_text(body['choices'][0]['message'].get('content') or '')
 return common.parse_response(body)
