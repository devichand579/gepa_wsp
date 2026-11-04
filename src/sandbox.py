"""Feedback-aware NumPy weight programs, inside an isolated no-network namespace."""
import ast,builtins,hashlib,json,resource
from pathlib import Path
import numpy as np

def parameter_gradient_reasons(tree):
 """Conservative dependency screen; selected programs still receive source review."""
 def names(node):
  result={n.id for n in ast.walk(node) if isinstance(n,ast.Name)}
  for n in ast.walk(node):
   if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='f' and isinstance(n.slice,ast.Constant):
    if n.slice.value in {'prediction','probs','logits','error','errors','residual','residuals','loss','losses','y','features','x'}:result.add('$'+n.slice.value)
  return result
 def roots(node):
  if isinstance(node,ast.Name):return {node.id}
  if isinstance(node,(ast.Subscript,ast.Attribute)):return roots(node.value)
  if isinstance(node,(ast.Tuple,ast.List)):return set().union(*(roots(n) for n in node.elts))
  return set()
 def returns(function):
  def visit(node):
   for child in ast.iter_child_nodes(node):
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Lambda)):continue
    if isinstance(child,ast.Return):yield child
    else:yield from visit(child)
  return list(visit(function))
 def returned_names(value):
  if isinstance(value,ast.Dict):
   result=set()
   for key,item in zip(value.keys,value.values):
    if isinstance(key,ast.Constant) and key.value=='weights':result.update(names(item))
   if not any(isinstance(key,ast.Constant) and key.value=='weights' for key in value.keys):
    for item in value.values:result.update(names(item))
   return result
  return names(value) if value is not None else set()
 deps={};spatial=set();aliases=set();sinks=['w']
 for function in (n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)):
  for node in returns(function):
   refs=returned_names(node.value);deps.setdefault(function.name,set()).update(refs)
   if function.name=='edit':sinks.extend(refs)
 for node in ast.walk(tree):
  if isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
   targets=node.targets if isinstance(node,ast.Assign) else [node.target]
   value=node.value
   if value is None:continue
   refs=names(value)
   for target in targets:
    for key in roots(target):deps.setdefault(key,set()).update(refs)
   if isinstance(value,ast.Subscript) and 'w' in roots(value):
    for target in targets:aliases.update(roots(target))
   if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='gradient' and value.args and 'f' in names(value.args[0]) and 'w' not in names(value.args[0]):
    for target in targets:spatial.update(roots(target))
 for name in aliases:deps.setdefault('w',set()).add(name)
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'append','extend','insert'}:
   for key in roots(node.func.value):deps.setdefault(key,set()).update(set().union(*(names(arg) for arg in node.args)))
 categorical_hessian=any(isinstance(node,ast.BinOp) and isinstance(node.op,ast.Sub) and isinstance(node.left,ast.Call) and ast.unparse(node.left.func) in {'np.diag','np.diagflat','numpy.diag'} and isinstance(node.right,ast.Call) and ast.unparse(node.right.func) in {'np.outer','numpy.outer'} for node in ast.walk(tree))
 seeds={name for name in set(deps)|set().union(*deps.values()) if (any(term in name.lower() for term in ['grad','hess','newton','deriv']) or name.lower() in {'dce','dce0','d2ce','d2ce0','g_w','g_b','gw','gb'} or (categorical_hessian and name.lower()=='g')) and name not in spatial}
 def ancestry(name):
  found=set();todo=[name]
  while todo:
   current=todo.pop()
   if current not in found:found.add(current);todo.extend(deps.get(current,set()))
  return found
 seeds={name for name in seeds if ancestry(name)&{'w','$prediction','$probs','$logits','$error','$errors','$residual','$residuals','$loss','$losses'}}
 tainted=set(seeds)
 while True:
  extra={name for name,refs in deps.items() if refs&tainted}
  if extra<=tainted:break
  tainted.update(extra)
 affected=set(sinks)&tainted
 # Detect residual-feature correlation steps even when no derivative name is used.
 # A closed-form solve is a barrier for this second screen, not for named gradients.
 def outside_solves(node):
  if isinstance(node,ast.Call) and ast.unparse(node.func) in {'np.linalg.solve','np.linalg.lstsq','np.linalg.pinv','numpy.linalg.solve','numpy.linalg.lstsq','numpy.linalg.pinv'}:return
  yield node
  for child in ast.iter_child_nodes(node):yield from outside_solves(child)
 def unsolved_names(node):
  result=set()
  for item in outside_solves(node):
   if isinstance(item,ast.Name):result.add(item.id)
  return result
 def references(node):
  refs=names(node);return set().union(*(ancestry(name) for name in refs)) if refs else set()
 def residual(node):
  refs=references(node)
  return bool(refs&{'$prediction','$probs','$error','$errors','$residual','$residuals'}) and '$y' in refs
 def features(node):return bool(references(node)&{'$features','$x','w'})
 correlations=set();unsolved={}
 for node in ast.walk(tree):
  if isinstance(node,ast.FunctionDef):
   for ret in returns(node):
    if ret.value is not None:unsolved.setdefault(node.name,set()).update(unsolved_names(ret.value))
  if not isinstance(node,(ast.Assign,ast.AnnAssign,ast.AugAssign)) or node.value is None:continue
  targets=node.targets if isinstance(node,ast.Assign) else [node.target]
  keys=set().union(*(roots(target) for target in targets))
  for key in keys:unsolved.setdefault(key,set()).update(unsolved_names(node.value))
  for item in outside_solves(node.value):
   pairs=[]
   if isinstance(item,ast.BinOp) and isinstance(item.op,(ast.Mult,ast.MatMult)):pairs=[(item.left,item.right)]
   if isinstance(item,ast.Call) and isinstance(item.func,ast.Attribute) and item.func.attr in {'dot','matmul','einsum','tensordot','multiply'}:
    args=item.args[1:] if item.func.attr=='einsum' else item.args
    if len(args)>=2:pairs=[(a,b) for i,a in enumerate(args) for b in args[i+1:]]
    elif len(args)==1:pairs=[(item.func.value,args[0])]
   if any((residual(a) and features(b)) or (residual(b) and features(a)) for a,b in pairs):correlations.update(keys)
 for name in aliases:unsolved.setdefault('w',set()).add(name)
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {'append','extend','insert'}:
   for key in roots(node.func.value):unsolved.setdefault(key,set()).update(set().union(*(unsolved_names(arg) for arg in node.args)))
 contaminated=set(correlations)
 while True:
  extra={name for name,refs in unsolved.items() if refs&contaminated}
  if extra<=contaminated:break
  contaminated.update(extra)
 affected|=set(sinks)&contaminated
 return ['Parameter-gradient/Newton or unsolved residual-feature correlation dependencies reach returned weights: '+', '.join(sorted(affected))] if affected else []

resource.setrlimit(resource.RLIMIT_CPU,(60,60))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
source=Path('/work/code.py').read_text();assert len(source)<=150000
bad_names={'open','eval','exec','compile','getattr','setattr','delattr','globals','locals','vars','input','help','breakpoint','memoryview','__import__'}
bad_attrs={'load','save','savez','savez_compressed','loadtxt','savetxt','fromfile','tofile','memmap','ctypeslib','ctypes','f2py','testing','distutils','core','_core','recfromcsv','genfromtxt','DataSource','dump','dumps'}
tree=ast.parse(source)
method_errors=parameter_gradient_reasons(tree)
if method_errors:raise ValueError('METHOD_VIOLATION: '+str(method_errors)+'. Use statistical construction or closed-form fitting, not parameter-gradient updates. Spatial input derivatives are permitted.')
for node in ast.walk(tree):
 if isinstance(node,ast.Import):
  assert all(a.name in {'numpy','numpy.linalg','numpy.lib.stride_tricks','math'} and not (a.asname or '').startswith('_') for a in node.names),'Only NumPy and math imports are available'
 if isinstance(node,ast.ImportFrom):
  assert node.level==0 and node.module in {'numpy','numpy.linalg','numpy.lib.stride_tricks','math'},'Only NumPy and math imports are available'
  assert all(a.name!='*' and not a.name.startswith('_') and a.name not in bad_attrs and not (a.asname or '').startswith('_') for a in node.names),'Unsafe import name'
 if isinstance(node,(ast.AsyncWith,ast.AsyncFunctionDef,ast.AsyncFor)):raise ValueError('Disallowed syntax: '+type(node).__name__)
 if isinstance(node,ast.Name) and (node.id in bad_names or node.id.startswith('__')):raise ValueError('Disallowed name: '+node.id)
 if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in bad_attrs):raise ValueError('Disallowed attribute: '+node.attr)
safe={k:getattr(builtins,k) for k in dir(builtins) if not k.startswith('_') and k not in bad_names|{'exit','quit','license','credits','copyright'}}
safe['__import__']=__import__;safe['__build_class__']=__build_class__
ns={'np':np,'__builtins__':safe,'__name__':'feedback_weight_program'};np.random.seed(0)
exec(compile(tree,'<feedback-weight-program>','exec'),ns)
assert callable(ns.get('edit')),'Define edit(w,f)'
with np.load('/work/weights.npz',allow_pickle=False) as z:original={k:z[k].copy() for k in z.files}
with np.load('/work/feedback.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
for v in f.values():v.setflags(write=False)
w={k:v.copy() for k,v in original.items()};result=ns['edit'](w,f);diagnostics={}
if isinstance(result,dict) and 'weights' in result:diagnostics=result.get('diagnostics',{});w=result['weights']
elif result is not None:w=result
allowed=json.loads(Path('/work/allowed.json').read_text());assert set(w)==set(original)
for k,v in w.items():
 v=np.asarray(v,dtype=np.float32);assert v.shape==original[k].shape,f'Wrong shape: {k}'
 assert np.isfinite(v).all() and np.max(np.abs(v),initial=0)<=1000,f'Nonfinite or oversized weights: {k}'
 if k not in allowed:assert np.array_equal(v,original[k]),f'Cannot edit {k}'
 w[k]=v
np.savez('/work/output.npz',**w)
def convert(value):
 if isinstance(value,np.ndarray):return value.tolist()
 if isinstance(value,np.generic):return value.item()
 raise TypeError(type(value).__name__)
diagnostics=dict(diagnostics) if isinstance(diagnostics,dict) else {'program_statistics':diagnostics}
diagnostics['execution_worker_sha256']=hashlib.sha256(Path('/worker.py').read_bytes()).hexdigest()
try:report=json.dumps(diagnostics,default=convert,allow_nan=False)
except (ValueError,TypeError):report=json.dumps({'diagnostic_error':'Return finite JSON-compatible diagnostics'})
if len(report)>24000:report=json.dumps({'diagnostic_error':'Diagnostic report exceeds 24000 characters; return a smaller summary'})
Path('/work/diagnostics.json').write_text(report)
print('VALID_FEEDBACK_WEIGHT_PROGRAM')
