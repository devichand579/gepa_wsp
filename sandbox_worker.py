"""Runs only inside a minimal bubblewrap namespace, never in the experiment host."""
import ast, builtins, json, resource
from pathlib import Path
import numpy as np

resource.setrlimit(resource.RLIMIT_CPU,(20,20))
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
source=Path('/work/code.py').read_text()
tree=ast.parse(source)
bad_names={'open','eval','exec','compile','getattr','setattr','delattr','globals','locals','vars','input','help','breakpoint','memoryview','__import__'}
bad_attrs={'load','save','savez','savez_compressed','loadtxt','savetxt','fromfile','tofile','memmap','ctypeslib','ctypes','f2py','testing','distutils','lib','core','_core','recfromcsv','genfromtxt','DataSource','dump','dumps','gradient','polyfit','lstsq'}
for node in ast.walk(tree):
    if isinstance(node,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.With,ast.AsyncWith,ast.AsyncFunctionDef,ast.Global,ast.Nonlocal)):
        raise ValueError('Disallowed syntax: '+type(node).__name__)
    if isinstance(node,ast.Name) and (node.id in bad_names or node.id.startswith('__')):
        raise ValueError('Disallowed name: '+node.id)
    if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in bad_attrs):
        raise ValueError('Disallowed attribute: '+node.attr)
    if isinstance(node,ast.AsyncFor):
        raise ValueError('Async loops are not allowed')
safe={k:getattr(builtins,k) for k in dir(builtins)
      if not k.startswith('_') and k not in bad_names|{'exit','quit','license','credits','copyright'}}
# NumPy's native dispatch may import its own helpers using the caller's builtins.
# Generated code cannot name this builtin: AST checks reject imports and dunders.
# The filesystem/network namespace independently excludes all private data.
safe['__import__']=__import__
ns={'np':np,'__builtins__':safe}
np.random.seed(0)
exec(compile(tree,'<tensor-edit>','exec'),ns)
assert callable(ns.get('edit')), 'Define edit(w)'
with np.load('/work/input.npz',allow_pickle=False) as z:
    original={k:z[k].copy() for k in z.files}
w={k:v.copy() for k,v in original.items()}
result=ns['edit'](w)
if result is not None: w=result
allowed=json.loads(Path('/work/allowed.json').read_text())
assert set(w)==set(original),'Tensor keys must not change'
for k,v in w.items():
    v=np.asarray(v,dtype=np.float32)
    assert v.shape==original[k].shape, f'Wrong shape for {k}'
    assert np.isfinite(v).all(),f'Nonfinite values in {k}'
    assert np.max(np.abs(v))<=1000,f'Weight magnitude exceeds 1000 in {k}'
    if k not in allowed: assert np.array_equal(v,original[k]),f'Cannot edit {k} in this module'
    w[k]=v
np.savez('/work/output.npz',**w)
print('VALID_TENSOR_EDIT')
