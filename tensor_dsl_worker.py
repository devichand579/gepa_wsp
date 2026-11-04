import json,resource
import numpy as np
from tensor_dsl import apply

resource.setrlimit(resource.RLIMIT_CPU,(20,20))
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
with open('/work/program.json') as f:program=json.load(f)
with open('/work/allowed.json') as f:allowed=json.load(f)
with np.load('/work/input.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
assert isinstance(program,dict) and isinstance(program.get('ops'),list)
assert 1<=len(program['ops'])<=256 and len(json.dumps(program))<=180000
result=state;errors=[];applied=0
for i,op in enumerate(program['ops']):
    try:
        result=apply(result,{'ops':[op]},allowed);applied+=1
    except (AssertionError,ValueError,TypeError,KeyError,IndexError,SyntaxError,OverflowError) as e:
        errors.append({'operation':i,'error':str(e)[:1200]})
with open('/work/status.json','w') as f:json.dump({'applied':applied,'rejected':errors,'total':len(program['ops'])},f)
assert applied>0,'No valid parameter operations'
np.savez('/work/output.npz',**result)
print('VALID_DECLARATIVE_WEIGHT_EDIT')
