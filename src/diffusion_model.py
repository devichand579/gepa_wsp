"""Small unconditional DDPM on 16x16 MNIST; no final-test access in preparation/training."""
import json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import core as common
ROOT=common.ROOT
STEPS=32

def schedule(device='cpu'):
 a=torch.tensor([math.cos((i/STEPS+.008)/1.008*math.pi/2)**2 for i in range(STEPS+1)],dtype=torch.float64,device=device)
 beta=(1-a[1:]/a[:-1]).clamp(max=.999).float();alpha=1-beta;abar=alpha.cumprod(0);prev=F.pad(abar[:-1],(1,0),value=1)
 return {'beta':beta,'alpha':alpha,'abar':abar,'prev':prev,'posterior_variance':beta*(1-prev)/(1-abar)}

class Denoiser(nn.Module):
 def __init__(self):
  super().__init__();self.conv1=nn.Conv2d(2,32,3,padding=1);self.conv2=nn.Conv2d(32,64,3,padding=1);self.conv3=nn.Conv2d(64,64,3,padding=1);self.out=nn.Conv2d(64,1,3,padding=1)
 def forward(self,x,t):
  time_channel=(t.float()/(STEPS-1))[:,None,None,None].expand(-1,1,16,16)
  h=F.relu(self.conv1(torch.cat([x,time_channel],1)));h=F.relu(self.conv2(h));h=F.relu(self.conv3(h));return self.out(h)

def initial(seed):
 torch.manual_seed(seed);return {k:v.detach().numpy().copy() for k,v in Denoiser().state_dict().items()}
def prepare(seed):
 folder=ROOT/'data/diffusion'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
 for split in ['feedback','selection']:
  d=common.load_npz(ROOT/'data/mnist'/f'seed{seed}'/(split+'.npz'))
  clean=F.interpolate(torch.from_numpy(d['x']),size=(16,16),mode='bilinear',align_corners=False)*2-1
  np.savez(folder/(split+'.npz'),clean=clean.numpy(),ids=d['ids'])
 common.dump(folder/'manifest.json',{'seed':seed,'source':'same MNIST train-partition IDs as classification','feedback':2000,'selection':1000,'resolution':16,'normalization':'bilinear resize, then 2*x-1','steps':STEPS,'schedule':'cosine alpha_bar(t)=cos((t+.008)/1.008*pi/2)^2, beta clipped at.999','parameter_count':sum(v.size for v in initial(seed).values()),'final_test':'official MNIST test; not loaded by this module','evaluation_noise_seeds':{'feedback':81000+seed,'selection':82000+seed,'final':83000+seed},'sampler':'DDPM posterior, predicted x0 clipped[-1,1],32 steps'})

def fixed_batch(clean,seed,device):
 generator=torch.Generator(device=device).manual_seed(seed);x=torch.from_numpy(clean).to(device)
 t=torch.randint(STEPS,(len(x),),device=device,generator=generator);eps=torch.randn(x.shape,device=device,generator=generator)
 a=schedule(device)['abar'][t,None,None,None];return a.sqrt()*x+(1-a).sqrt()*eps,t,eps

def evaluate(state,clean,noise_seed,device):
 net=Denoiser().to(device);net.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()});net.eval();x,t,eps=fixed_batch(clean,noise_seed,device);losses=[]
 with torch.inference_mode():
  for ids in torch.arange(len(x),device=device).split(128):losses.append((net(x[ids],t[ids])-eps[ids]).square().flatten(1).mean(1).cpu().numpy())
 loss=np.concatenate(losses);return {'mse':float(loss.mean()),'per_example_mse':loss,'score':-float(loss.mean())}

def sample(state,n,seed,device='cuda:1'):
 net=Denoiser().to(device);net.load_state_dict({k:torch.from_numpy(v) for k,v in state.items()});net.eval();s=schedule(device);g=torch.Generator(device=device).manual_seed(seed);x=torch.randn((n,1,16,16),device=device,generator=g)
 with torch.inference_mode():
  for step in reversed(range(STEPS)):
   t=torch.full((n,),step,device=device,dtype=torch.long);eps=net(x,t);a=s['abar'][step];prev=s['prev'][step];beta=s['beta'][step]
   x0=((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
   mean=beta*prev.sqrt()/(1-a)*x0+(1-prev)*s['alpha'][step].sqrt()/(1-a)*x
   x=mean+(s['posterior_variance'][step].sqrt()*torch.randn(x.shape,device=device,generator=g) if step else 0)
 return x.cpu().numpy()

def adam(seed,device,steps=1000):
 prepare(seed);data=ROOT/'data/diffusion'/f'seed{seed}';fb=common.load_npz(data/'feedback.npz');sv=common.load_npz(data/'selection.npz');folder=ROOT/'baselines/diffusion'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'adam_result.json').exists():return
 x=torch.from_numpy(fb['clean']).to(device);s=schedule(device);best=float('inf');history=[];start=time.time()
 for lr in [1e-3,3e-4]:
  net=Denoiser().to(device);net.load_state_dict({k:torch.from_numpy(v) for k,v in initial(seed).items()});opt=torch.optim.Adam(net.parameters(),lr=lr);g=torch.Generator(device=device).manual_seed(84000+seed)
  for step in range(1,steps+1):
   ids=torch.randint(len(x),(64,),device=device,generator=g);t=torch.randint(STEPS,(64,),device=device,generator=g);eps=torch.randn(x[ids].shape,device=device,generator=g);a=s['abar'][t,None,None,None]
   noisy=a.sqrt()*x[ids]+(1-a).sqrt()*eps;opt.zero_grad(set_to_none=True);loss=(net(noisy,t)-eps).square().mean();loss.backward();opt.step()
   if step%100==0 or step==steps:
    state={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()};result=evaluate(state,sv['clean'],82000+seed,device);row={'lr':lr,'step':step,'mse':result['mse']};history.append(row)
    if row['mse']<best:best=row['mse'];info=row;np.savez(folder/'adam_best.npz',**state)
  print(json.dumps({'seed':seed,'lr':lr,'best_selection_mse':best}),flush=True)
 common.dump(folder/'adam_result.json',{'seed':seed,'best':info,'history':history,'steps_per_lr':steps,'total_optimizer_steps':steps*2,'wall_seconds':time.time()-start,'final_test_evaluated':False})
