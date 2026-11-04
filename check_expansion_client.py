import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import expansion_client as client
import experiment as common
original=common.MODEL
client.configure('nemotron');assert common.MODEL==client.MODELS['nemotron']['id'];assert common.PRICES[common.MODEL]==(.05,.20)
client.configure('deepseek');assert common.MODEL==client.MODELS['deepseek']['id']
with tempfile.TemporaryDirectory() as td:
 folder=Path(td);response={'choices':[{'finish_reason':'stop','message':{'content':'```python\ndef edit(w,f):\n    return w\n```'}}]}
 # Saved provider failures cannot create another paid request.
 common.dump(folder/'service_failure.json',{'ledger_call':-1})
 with patch.object(common.Budget,'reserve',side_effect=AssertionError('Paid request forbidden')):
  try:client.call('unused',folder,'test')
  except RuntimeError as e:assert 'not retried' in str(e)
  else:raise AssertionError('Failure replay dispatched')
assert np.allclose(client.MODELS['minimax']['prices'],[.15,.60])
print('PASS: per-process model/pricing selection; recorded failures cannot redispatch; no paid calls in this check')
