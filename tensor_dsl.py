"""Declarative, bounded parameter edits: no Python programs, data, or model calls."""
from __future__ import annotations
import ast
import json
import numpy as np

FUNCTIONS={k:getattr(np,k) for k in ['sin','cos','exp','tanh','sqrt','abs','maximum','minimum','where','clip','sign']}
NODES=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,ast.Constant,
       ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod,ast.FloorDiv,ast.USub,ast.UAdd,
       ast.BitAnd,ast.BitOr,ast.Invert,ast.Compare,ast.Eq,ast.NotEq,ast.Lt,ast.LtE,ast.Gt,ast.GtE)

def index(spec,shape):
    if spec is None:return tuple(slice(None) for _ in shape)
    assert isinstance(spec,list) and len(spec)<=len(shape),'Index must have at most one entry per tensor dimension'
    result=[]
    for item,size in zip(spec,shape):
        if item==':':result.append(slice(None))
        elif isinstance(item,int):
            assert -size<=item<size,'Index out of range';result.append(item)
        elif isinstance(item,list) and 2<=len(item)<=3:
            assert all(v is None or isinstance(v,int) for v in item)
            assert len(item)<3 or item[2]!=0
            result.append(slice(*item))
        else:raise ValueError('Use integer, ":", or [start,stop,step] for an index')
    result.extend(slice(None) for _ in range(len(shape)-len(result)))
    return tuple(result)

def formula(expression,shape):
    assert isinstance(expression,str) and len(expression)<=4000,'Formula must be a string with at most 4000 characters'
    tree=ast.parse(expression,mode='eval');nodes=list(ast.walk(tree));assert len(nodes)<=700,'Formula exceeds 700 AST nodes'
    names=set(FUNCTIONS)|{'pi','e'}|{prefix+str(i) for prefix in ['i','x','n'] for i in range(len(shape))}
    for node in nodes:
        assert isinstance(node,NODES),f'Forbidden formula syntax: {type(node).__name__}'
        if isinstance(node,ast.Name):assert node.id in names,f'Unknown formula symbol: {node.id}'
        if isinstance(node,ast.Call):
            assert isinstance(node.func,ast.Name) and node.func.id in FUNCTIONS
            assert not node.keywords and len(node.args)<=4,'Formula functions accept positional arguments only, at most four'
        if isinstance(node,ast.Constant):
            assert isinstance(node.value,(int,float,bool)) and abs(node.value)<=1e6,'Formula constants must be numeric with absolute value <=1e6'
            if not isinstance(node.value,bool):node.value=float(node.value)
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            power=node.right
            if isinstance(power,ast.UnaryOp) and isinstance(power.op,(ast.USub,ast.UAdd)):power=power.operand
            assert isinstance(power,ast.Constant) and isinstance(power.value,(int,float)) and abs(power.value)<=16,'Use a literal power between -16 and 16'
    env=dict(FUNCTIONS,pi=np.pi,e=np.e)
    for i,size in enumerate(shape):
        coords=np.arange(size,dtype=np.float64).reshape(tuple(size if j==i else 1 for j in range(len(shape))))
        env['i'+str(i)]=coords;env['x'+str(i)]=2*coords/(size-1)-1 if size>1 else coords*0;env['n'+str(i)]=float(size)
    with np.errstate(all='ignore'):
        value=eval(compile(tree,'<coordinate-formula>','eval'),{'__builtins__':{}},env)
    return np.broadcast_to(np.asarray(value,dtype=np.float64),shape)

def shifted(a,offsets):
    assert len(offsets)==a.ndim and all(isinstance(v,int) for v in offsets),f'Shift needs {a.ndim} integer offsets for the source rank; use source_reshape before a spatial shift'
    result=np.zeros_like(a);src=[];dst=[]
    for d,n in zip(offsets,a.shape):
        if abs(d)>=n:return result
        src.append(slice(max(0,-d),min(n,n-d)));dst.append(slice(max(0,d),min(n,n+d)))
    result[tuple(dst)]=a[tuple(src)];return result

def apply(state,program,allowed=None):
    assert isinstance(program,dict) and isinstance(program.get('ops'),list)
    assert 1<=len(program['ops'])<=256,'Require 1–256 operations'
    assert len(json.dumps(program))<=180000,'Program too large'
    w={k:v.copy() for k,v in state.items()};allowed=set(w if allowed is None else allowed)
    for op in program['ops']:
        kind=op['op'];name=op['tensor'];assert name in w and name in allowed,f'Cannot edit {name}'
        ix=index(op.get('index'),w[name].shape)
        original_shape=np.asarray(w[name][ix]).shape
        view=np.asarray(w[name][ix],dtype=np.float64).copy()
        if 'reshape' in op:
            assert isinstance(op['reshape'],list) and all(isinstance(x,int) and x>0 for x in op['reshape']),'Target reshape must list positive integer dimensions'
            view=view.reshape(op['reshape'])
        if kind in ['set','add','scale']:
            assert ('value' in op)!=('expr' in op),'Specify value or expr'
            value=formula(op['expr'],view.shape) if 'expr' in op else np.asarray(op['value'],dtype=np.float64)
            value=np.broadcast_to(value,view.shape)
            if kind=='set':view=value.copy()
            elif kind=='add':view+=value
            else:view*=value
        elif kind=='scatter':
            coordinates=np.asarray(op['coordinates'])
            assert coordinates.ndim==2 and coordinates.shape[1]==view.ndim and len(coordinates)<=4096,'Scatter needs up to 4096 full-rank integer coordinates'
            assert np.issubdtype(coordinates.dtype,np.integer),'Scatter coordinates must be integers'
            for axis,size in enumerate(view.shape):
                assert ((coordinates[:,axis]>=-size)&(coordinates[:,axis]<size)).all(),'Scatter coordinate outside selected view'
            values=np.broadcast_to(np.asarray(op['value'],dtype=np.float64),(len(coordinates),))
            view[tuple(coordinates.T)]=values
        elif kind=='copy':
            source_name=op['source_tensor'];assert source_name in w,'Unknown source tensor'
            source=np.asarray(w[source_name][index(op.get('source_index'),w[source_name].shape)],dtype=np.float64).copy()
            if 'source_reshape' in op:source=source.reshape(op['source_reshape'])
            if 'transpose' in op:source=source.transpose(op['transpose'])
            if 'shift' in op:source=shifted(source,op['shift'])
            if source.size==view.size:source=source.reshape(view.shape)
            source=np.broadcast_to(source,view.shape)
            view=source*float(op.get('factor',1))+float(op.get('offset',0))
        elif kind=='normalize':
            axes=op.get('axes',list(range(view.ndim)))
            if isinstance(axes,int):axes=[axes]
            assert isinstance(axes,list) and all(isinstance(a,int) and -view.ndim<=a<view.ndim for a in axes),'Normalization axes must refer to the selected/reshaped view dimensions'
            axes=tuple(axes)
            if op.get('center',False):view-=view.mean(axis=axes,keepdims=True)
            norm=np.sqrt(np.sum(view*view,axis=axes,keepdims=True))
            view*=float(op.get('norm',1))/np.maximum(norm,1e-8)
        else:raise ValueError('Unknown tensor operation: '+str(kind))
        assert np.isfinite(view).all(),'Nonfinite weight result'
        assert np.max(np.abs(view),initial=0)<=1000,'Weight magnitude exceeds 1000'
        w[name][ix]=view.reshape(original_shape).astype(np.float32)
    assert set(w)==set(state)
    for k,v in w.items():
        assert v.shape==state[k].shape and np.isfinite(v).all()
        if k not in allowed:assert np.array_equal(v,state[k])
    return w

SPEC='''Return ONE JSON object with keys "hypothesis" (short text), "prediction" (short text), and "ops" (1–256 operations). Keep the complete object under 180,000 characters. All resulting tensor values must be finite float32 with absolute value <=1000; names and shapes cannot change.
This is direct parameter editing. No executable Python, loops, reducers over examples, network/layer execution, regression, covariance fitting, prototype-feature fitting, gradients, optimizers or candidate search can be expressed in this interface.
Each op has "op" and "tensor", optional "index" and "reshape":
- "set", "add", "scale": supply either numeric "value" (scalar or nested array) OR coordinate "expr".
- "scatter": "coordinates" is a list of full-rank integer coordinates within the selected/reshaped parameter view, and "value" is a scalar or one value per coordinate. This sets sparse coefficients without listing background zeros.
- "copy": "source_tensor", optional "source_index", "source_reshape", "transpose" axes, zero-padded "shift" offsets, "factor" and "offset". The source is an existing parameter array only. Equal-size arrays reshape to the target; otherwise broadcasting applies.
- "normalize": "axes" (integer or list), optional "center" boolean and "norm" target L2 norm. This normalizes weights, never data.
Index is a list, one item per dimension: an integer, ":", or [start,stop,step] with Python-exclusive stop and null for an open bound. Omitted dimensions select all.
Optional reshape applies to the selected target view while computing the operation, then writes back in its original shape.
Coordinate expr may reference only i0,i1,... (indices of selected/reshaped view), x0,x1,... (coordinates normalized to [-1,1]), n0,n1,... (axis sizes), pi,e, numeric constants, arithmetic +,-,*,/,**,%,//, comparisons and &,|; functions sin,cos,exp,tanh,sqrt,abs,maximum,minimum,where,clip,sign. The exponent of ** must be a literal between -16 and 16; numeric constants must have absolute value <=1e6. No array/tensor reads, indexing, attributes or arbitrary functions inside expr.
Examples of syntax only:
{"op":"set","tensor":"fc1.bias","value":0}
{"op":"set","tensor":"fc1.weight","index":[0,":"],"reshape":[28,28],"expr":"exp(-((x0/0.6)**2+(x1/0.4)**2))"}
{"op":"normalize","tensor":"fc1.weight","axes":1,"norm":1}
{"op":"copy","tensor":"fc1.weight","index":[1,":"],"source_tensor":"fc1.weight","source_index":[0,":"],"source_reshape":[28,28],"shift":[1,0]}
Examples demonstrate syntax, not a recommended classification strategy. Choose the computations and all coefficients yourself using task knowledge, supplied feedback and parameter diagnostics. Only the listed parameter operations execute; all network evaluation is performed by the harness.
'''
