"""Fixed SST-2 transformer container and its Adam baseline."""
from __future__ import annotations


import json, types


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


import core as image


ROOT=image.ROOT
VOCAB=None

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.norm1=nn.LayerNorm(32);self.qkv=nn.Linear(32,96);self.proj=nn.Linear(32,32)
        self.norm2=nn.LayerNorm(32);self.ff1=nn.Linear(32,64);self.ff2=nn.Linear(64,32)
    def forward(self,x,mask):
        b,l,d=x.shape;q,k,v=self.qkv(self.norm1(x)).chunk(3,dim=-1)
        q,k,v=[z.reshape(b,l,2,16).transpose(1,2) for z in [q,k,v]]
        logits=(q@k.transpose(-1,-2))/4
        logits=logits.masked_fill(~mask[:,None,None,:],-1e4)
        a=(logits.softmax(-1)@v).transpose(1,2).reshape(b,l,32)
        x=x+self.proj(a);return x+self.ff2(F.relu(self.ff1(self.norm2(x))))


class Network(nn.Module):
    def __init__(self,task='sst2'):
        super().__init__();self.embed=nn.Embedding(2000,32,padding_idx=0);self.pos=nn.Embedding(64,32)
        self.block0=Block();self.block1=Block();self.norm=nn.LayerNorm(32);self.head=nn.Linear(32,2)
    def forward(self,ids,trace=False):
        mask=ids.ne(0);h=self.embed(ids)+self.pos(torch.arange(ids.shape[1],device=ids.device))[None,:,:]
        h=self.block1(self.block0(h,mask),mask);h=self.norm(h)
        features=(h*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        out=self.head(features);return (out,features) if trace else out


def initial(task,seed):
    torch.manual_seed(seed);net=Network();return {k:v.detach().numpy().copy() for k,v in net.state_dict().items()}


def modules(task):return ['embed','pos','block0','block1','norm','head']


def spec(task,state):
    return {'task':'SST-2 binary sentiment','objective':'Maximize held-out classification accuracy',
            'input':'int64 [N,64] token IDs; lowercase regex tokenizer, truncate right, pad right with 0; unknown token 1',
            'class_indices':['negative','positive'],'tensor_shapes':{k:list(v.shape) for k,v in state.items()},
            'parameter_count':sum(v.size for v in state.values()),
            'vocabulary_id_to_word':VOCAB['words'],
            'vocabulary_provenance':'2000 tokens including PAD/UNK; frequency fit on feedback texts only; no sentiment aggregates or pretrained embeddings',
            'forward':'h=embed(ids)+pos(arange(64)); for block in [block0,block1]: z=norm1(h); q,k,v=linear(z,qkv).chunk(3); attention=softmax(q@k.T/sqrt(16), excluding padding keys)@v in 2 heads; h=h+linear(attention,proj); h=h+linear(relu(linear(norm2(h),ff1)),ff2); h=norm(h); pooled=masked_mean(h,ids!=0); logits=linear(pooled,head); predict argmax',
            'conventions':'width32; bidirectional attention, 2 heads of width16; no dropout; pre-LayerNorm, epsilon1e-5; qkv output order q then k then v; PyTorch linear weights [out,in]; final mean excludes padding; all parameters untrained before construction'}


def decode(ids):return ' '.join(VOCAB['words'][int(i)] for i in ids if i!=0)


def example_packet(x,y,n=16):
    chosen=[]
    for label in [0,1]:chosen.extend(np.flatnonzero(y==label)[:n//2].tolist())
    return [{'feedback_index':i,'label':int(y[i]),'tokenized_text':decode(x[i])} for i in chosen]


def evaluate_state(task,state,x,y,device,features=False):
    net=Network().to(device);net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in state.items()});net.eval()
    probs=[];feats=[]
    with torch.inference_mode():
        for start in range(0,len(x),128):
            logits,h=net(torch.from_numpy(x[start:start+128]).to(device),trace=True)
            probs.append(logits.softmax(-1).cpu().numpy())
            if features:feats.append(h.cpu().numpy())
    p=np.concatenate(probs);pred=p.argmax(-1);truep=p[np.arange(len(y)),y];scores=.9*(pred==y)+.1*truep
    return {'accuracy':float((pred==y).mean()),'cross_entropy':float(-np.log(np.maximum(truep,1e-30)).mean()),'score':float(scores.mean()),
            'probs':p,'pred':pred,'scores':scores,'features':np.concatenate(feats) if features else None}


def adam(seed,device,folder):
    # The shared Adam function resolves its globals in a private dictionary.
    backend=install(seed);namespace=dict(vars(backend));namespace['run_adam']=image.run_adam
    fn=types.FunctionType(image.run_adam.__code__,namespace,'run_adam',image.run_adam.__defaults__)
    return fn('sst2',seed,device,folder,50)


def install(seed):
 global VOCAB
 VOCAB=json.loads((ROOT/"data/sst2"/f"seed{seed}"/"vocab.json").read_text())
 backend=types.SimpleNamespace(**vars(image))
 for name in ["Network","initial","modules","spec","example_packet","evaluate_state"]:
  setattr(backend,name,globals()[name])
 return backend
