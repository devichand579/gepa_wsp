import json,tempfile
from pathlib import Path
import experiment as common
import strict_experiment as strict
with tempfile.TemporaryDirectory() as td:
 p=Path(td);d=p/'llm/mutation_001';d.mkdir(parents=True)
 common.dump(d/'dispatch.json',{'ledger_call':10});common.dump(d/'meta.json',{'ledger_call':10})
 d=p/'llm/mutation_002';d.mkdir();common.dump(d/'dispatch.json',{'ledger_call':11})
 common.dump(d/'service_failure.json',{'ledger_call':11})
 assert common.mutation_api_calls(p)==2,'Cached response or failure counted incorrectly'
 before=len(json.loads((common.ROOT/'runtime/usage.json').read_text())['calls'])
 try:strict.call('unused',d,'must-not-dispatch')
 except RuntimeError as e:assert 'not retried' in str(e)
 else:raise AssertionError('Failed dispatch retried')
 assert len(json.loads((common.ROOT/'runtime/usage.json').read_text())['calls'])==before
print('PASS: successes/failures counted once; recorded provider failure cannot redispatch')
