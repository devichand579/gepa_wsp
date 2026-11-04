import numpy as np,torch
import diffusion_core as d
s=d.schedule();assert torch.all((s['beta']>0)&(s['beta']<1));assert torch.all(s['abar'][1:]<s['abar'][:-1]);assert s['posterior_variance'][0]==0
w=d.initial(0);assert sum(v.size for v in w.values())==56609
x=np.zeros((4,1,16,16),dtype=np.float32);a,t,eps=d.fixed_batch(x,42,'cpu');b,t2,eps2=d.fixed_batch(x,42,'cpu');assert torch.equal(a,b) and torch.equal(t,t2) and torch.equal(eps,eps2)
net=d.Denoiser();assert net(a,t).shape==a.shape
result=d.evaluate(w,x,42,'cpu');assert np.isfinite(result['mse'])
one=d.sample(w,4,42,'cpu');two=d.sample(w,4,42,'cpu');assert np.array_equal(one,two) and np.isfinite(one).all() and np.max(np.abs(one))<=1.00001
print('PASS: cosine schedule,56609 parameters, paired reproducible noising, denoising shape/loss, deterministic finite32-step sampler; no final-test reads')
