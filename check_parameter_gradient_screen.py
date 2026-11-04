"""Check the real flagged programs and preserve permitted statistical operations."""
import ast
from pathlib import Path
worker=ast.parse(Path('rich_worker.py').read_text());function=next(n for n in worker.body if isinstance(n,ast.FunctionDef) and n.name=='parameter_gradient_reasons');ns={'ast':ast};exec(compile(ast.Module(body=[function],type_ignores=[]),'<trusted-method-screen>','exec'),ns);screen=ns['parameter_gradient_reasons']
blocked=[Path('expansion_runs/classification/deepseek60/mnist/seed1/llm/mutation_024/edit.py'),Path('expansion_runs/classification/deepseek60/mnist/seed2/llm/mutation_018/edit.py'),Path('diffusion_runs/models/nemotron/seed0/construction.py')]
blocked.extend([Path('expansion_runs/classification/deepseek60/mnist/seed0/llm/mutation_012/edit.py'),Path('expansion_runs/classification/deepseek60/fashion/seed1/llm/mutation_016/edit.py'),Path('expansion_runs/classification/deepseek60/fashion/seed2/llm/mutation_031/edit.py'),Path('diffusion_runs/models/nemotron/seed1/llm/mutation_007/edit.py'),Path('diffusion_runs/models/nemotron/seed2/llm/mutation_011/edit.py')])
blocked.append(Path('diffusion_runs/repaired/models/nemotron/seed0/llm/mutation_004/edit.py'))
for p in blocked:assert screen(ast.parse(p.read_text())),str(p)
allowed=['''def edit(w,f):
 X=f['features']; Y=f['y']; w['head.weight']=np.linalg.solve(X.T@X+np.eye(X.shape[1]),X.T@Y).T
 return w
''','''def edit(w,f):
 grad=np.gradient(f['x'],axis=-1)
 w['bias'][:]=np.mean(grad)
 return w
''','''def edit(w,f):
 grad=f['x'].mean()
 return {'weights':w,'diagnostics':{'grad':float(grad)}}
''','''def edit(w,f):
 grad=f['features'][f['y']==1].mean(0)-f['features'][f['y']==0].mean(0)
 w['weight'][:]=grad
 return w
''','''def diagnostic(f):
 grad=f['prediction'].mean()
 return grad
def edit(w,f):
 return {'weights':w,'diagnostics':{'value':float(diagnostic(f))}}
''']
for code in allowed:assert not screen(ast.parse(code)),code
assert not screen(ast.parse("def edit(w,f):\n X=f['features']; r=f['prediction']-f['y']; b=X.T@r; correction=np.linalg.solve(X.T@X,b); w['weight']-=correction; return w"))
assert screen(ast.parse("def edit(w,f):\n X=f['features']; e=f['prediction']-f['y']; step=np.mean(e[:,None]*X,axis=0); w['weight']-=.01*step; return w"))
assert screen(ast.parse("def edit(w,f):\n grad=f['error'].mean(); W=w['bias']; W-=.01*grad; return w"))
assert screen(ast.parse("def edit(w,f):\n grad=f['prediction'].mean(); w['bias']-=.01*grad"))
print('PASS: all9 actual named/unnamed gradient/Newton programs rejected; direct and residual ridge fits, input derivatives and diagnostic-only statistics preserved')
