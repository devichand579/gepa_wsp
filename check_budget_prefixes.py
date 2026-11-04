"""Ensure budget extraction cannot choose a later, better checkpoint."""
import tempfile,json
from pathlib import Path
from classification_followup import select_prefix
with tempfile.TemporaryDirectory() as td:
 p=Path(td)
 for n,t in [(1,10),(2,20),(3,30)]:
  d=p/'llm'/f'mutation_{n:03d}';d.mkdir(parents=True);(d/'dispatch.json').write_text(json.dumps({'ledger_call':n,'time':t}))
 rows=[{'split':'selection','candidate':str(i),'time':t,'error':None,'metrics':{'score':s,'accuracy':s}} for i,t,s in [(0,1,.1),(1,15,.4),(2,25,.3),(3,35,.9)]]
 (p/'evaluations.jsonl').write_text('\n'.join(json.dumps(r) for r in rows));(p/'search_result.json').write_text('{}')
 assert select_prefix(p,0)['candidate']=='0'
 assert select_prefix(p,1)['candidate']=='1'
 assert select_prefix(p,2)['candidate']=='1','Later high-scoring checkpoint leaked into prefix'
 assert select_prefix(p,3)['candidate']=='3'
print('PASS: nested budget selection excludes future candidates and retains the best earlier selection score')
