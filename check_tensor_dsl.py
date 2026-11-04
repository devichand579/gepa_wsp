import numpy as np
from tensor_dsl import apply,formula

s={'a':np.zeros((2,4),dtype=np.float32),'b':np.ones(2,dtype=np.float32)}
p={'ops':[{'op':'set','tensor':'a','expr':'i0*10+i1'},
          {'op':'copy','tensor':'a','index':[1,':'],'source_tensor':'a','source_index':[0,':'],'source_reshape':[2,2],'shift':[0,1]}]}
r=apply(s,p,['a']);assert np.array_equal(r['a'][0],[0,1,2,3]);assert np.array_equal(r['a'][1],[0,0,0,2]);assert np.all(r['b']==1)
r=apply(s,{'ops':[{'op':'set','tensor':'a','value':1},{'op':'normalize','tensor':'a','axes':1}]})
assert np.allclose(np.linalg.norm(r['a'],axis=1),1)
for expr in ['__import__("os")','np.load("secret")','i0.__class__','sum(i0)','i0[0]','open("secret")','[1,2,3]']:
    try:formula(expr,(2,4))
    except (AssertionError,ValueError,SyntaxError):pass
    else:raise AssertionError('Forbidden expression accepted: '+expr)
try:apply(s,{'ops':[{'op':'set','tensor':'b','value':0}]},['a'])
except AssertionError:pass
else:raise AssertionError('Module constraint bypassed')
try:apply(s,{'ops':[{'op':'set','tensor':'a','expr':'exp(10000)'}]})
except AssertionError:pass
else:raise AssertionError('Nonfinite values accepted')
assert np.all(s['a']==0),'Input state mutated'
r=apply(s,{'ops':[{'op':'scatter','tensor':'a','coordinates':[[0,1],[1,3]],'value':[2,7]}]})
assert r['a'][0,1]==2 and r['a'][1,3]==7 and np.count_nonzero(r['a'])==2
print('PASS: coordinate construction, indexed copies/shifts, normalization, immutable inputs, module boundary, forbidden code/data access, numerical constraints')
