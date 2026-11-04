"""Independent SST-2 backend for the frozen strict GEPA engine.

Backend substitution is confined to this process. The image harness and its frozen
source files are unchanged. No pretrained target weights or tokenizers are used.
"""
from __future__ import annotations
import argparse,hashlib,json,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gepa.core.adapter import EvaluationBatch
import experiment as image
import strict_experiment as strict
from tensor_dsl import SPEC
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
def language_spec(task):
    examples=[{'op':'set','tensor':'pos.weight','value':0},
              {'op':'scatter','tensor':'embed.weight','coordinates':[[2,0],[3,1]],'value':[.1,.2]},
              {'op':'set','tensor':'block0.proj.weight','expr':'0.01*(i0==i1)'},
              {'op':'set','tensor':'head.weight','index':[0,':'],'expr':'0.01*x0'}]
    return SPEC.split('Examples of syntax only:')[0]+'\nIndices select the original tensor before optional reshape. No subsequent training occurs. Invalid operations are rejected individually; valid operations still execute; no repairs or free retries. Prefer compact formulas and sparse scatter assignments to long arrays.\nSyntax-only examples for this architecture, not a recommended sentiment strategy:\n'+json.dumps(examples)
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

class Adapter(strict.Adapter):
    def evaluate(self,batch,candidate,capture_traces=False):
        split=batch[0]['split'];ids=np.array([b['index'] for b in batch]);self.eval_calls+=len(batch)
        assert all(b['split']==split for b in batch)
        if capture_traces:assert split=='feedback'
        x=self.x[split][ids];y=self.y[split][ids];h=self.compiler.key(candidate)
        try:
            h,state=self.compiler.materialize(candidate);r=evaluate_state('sst2',state,x,y,self.device,capture_traces);error=None
        except Exception as e:
            error=str(e)[-4000:];r={'accuracy':0.,'cross_entropy':float('inf'),'score':-1.,'scores':np.full(len(batch),-1.),'pred':np.full(len(batch),-1),'probs':np.zeros((len(batch),2)),'features':None}
        image.append(self.folder/'evaluations.jsonl',{'candidate':h,'split':split,'n':len(batch),'capture_traces':capture_traces,'metrics':image.summary(r),'error':error})
        if split=='feedback':
            diag=self.compiler.folder/f'{h}.diagnostics.json'
            for record in self.recent:
                if record.get('child')==h:
                    record.update(measured_feedback_metrics=image.summary(r),execution_error=error)
                    if diag.exists():record['operation_diagnostics']=json.loads(diag.read_text())
        traj=None
        if capture_traces:
            traj=[{'split':split,'feedback_index':int(i),'label':int(label),'prediction':int(pred),'probs':p.round(4).tolist(),'error':error}
                  for i,label,pred,p in zip(ids,y,r['pred'],r['probs'])]
            self.latest={'r':r,'ids':ids,'state':None if error else state,'error':error,'h':h}
        return EvaluationBatch(outputs=[int(v) for v in r['pred']],scores=r['scores'].tolist(),trajectories=traj)
    def make_reflective_dataset(self,candidate,eval_batch,components_to_update):
        assert all(t['split']=='feedback' for t in eval_batch.trajectories)
        records=list(eval_batch.trajectories);errors=[r for r in records if r['prediction']!=r['label']]
        chosen=[]
        for label in [0,1]:chosen.extend([r for r in errors if r['label']==label][:4])
        for record in records:
            if len(chosen)>=12:break
            if record not in chosen:chosen.append(record)
        for record in chosen:record['tokenized_text']=decode(self.x['feedback'][record['feedback_index']])
        latest=self.latest
        feedback={'feedback_summary':image.summary(latest['r']),'examples':chosen,'error':latest['error'],
                  'weight_statistics':image.weight_summary(latest['state']) if latest['state'] else None,'selected_modules':components_to_update}
        feat=latest['r']['features']
        if feat is not None:feedback['feature_statistics']={'mean':feat.mean(0).round(4).tolist(),'std':feat.std(0).round(4).tolist(),'max':float(feat.max())}
        return {m:[feedback] for m in components_to_update}

def install(seed):
    global VOCAB
    VOCAB=json.loads((ROOT/'data/sst2'/f'seed{seed}'/'vocab.json').read_text())
    backend=types.SimpleNamespace(**vars(image))
    for name in ['Network','initial','modules','spec','example_packet','evaluate_state']:setattr(backend,name,globals()[name])
    strict.common=backend;strict.Adapter=Adapter;strict.language_spec=language_spec
    return backend

def adam(seed,device,folder):
    # The shared Adam function resolves its globals in a private dictionary.
    backend=install(seed);namespace=dict(vars(backend));namespace['run_adam']=image.run_adam
    fn=types.FunctionType(image.run_adam.__code__,namespace,'run_adam',image.run_adam.__defaults__)
    return fn('sst2',seed,device,folder,50)
def run(seed,device,proposals):
    install(seed);folder=ROOT/'text_runs/sst2'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    image.dump(folder/'text_config.json',{'seed':seed,'architecture':spec('sst2',initial('sst2',seed)),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initial_examples':16,
        'note':'Private process backend; no image source or primary run configuration modified.'})
    return strict.run('sst2',seed,device,folder,proposals)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--method',choices=['adam','search'],required=True);p.add_argument('--proposals',type=int,default=40);a=p.parse_args()
    torch.set_num_threads(2)
    if a.method=='adam':adam(a.seed,a.device,ROOT/'text_runs/sst2'/f'seed{a.seed}')
    else:run(a.seed,a.device,a.proposals)
