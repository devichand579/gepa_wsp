"""Shared networks, metrics, budget accounting and adapter state."""
from __future__ import annotations


import ast, fcntl, hashlib, json, os, re, time


from pathlib import Path


import numpy as np


import torch


from torch import nn


from torch.nn import functional as F


ROOT=Path(os.environ.get('GEPA_WSP_WORKSPACE',Path(__file__).resolve().parents[1]/'work'))
MODEL='accounts/fireworks/models/deepseek-v4p1-flash'
PRICES={MODEL:(.22,.66)}

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)


def append(path,obj):
    with open(path,'a') as f:f.write(json.dumps(obj,default=str)+'\n')


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


class Network(nn.Module):
    def __init__(self,task):
        super().__init__();self.task=task
        if task=='mnist':self.fc1=nn.Linear(784,128);self.fc2=nn.Linear(128,10)
        else:self.conv1=nn.Conv2d(1,16,3,padding=1);self.conv2=nn.Conv2d(16,32,3,padding=1);self.head=nn.Linear(1568,10)
    def forward(self,x,trace=False):
        if self.task=='mnist':
            h=F.relu(self.fc1(x.flatten(1)));out=self.fc2(h);features=h
        else:
            h=F.max_pool2d(F.relu(self.conv1(x)),2)
            h=F.max_pool2d(F.relu(self.conv2(h)),2)
            features=h.flatten(1);out=self.head(features)
        return (out,features) if trace else out


def initial(task,seed):
    torch.manual_seed(seed)
    net=Network(task)
    return {k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}


def modules(task):return ['fc1','fc2'] if task=='mnist' else ['conv1','conv2','head']


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def ascii_image(x):
    x=np.asarray(x).reshape(28,28);levels=' .:-=+*#%@'
    return '\n'.join(''.join(levels[min(9,int(v*10))] for v in row) for row in x)


def example_packet(x,y,n=10):
    chosen=[]
    for label in range(10):
        ids=np.flatnonzero(y==label)
        if len(ids):chosen.append(int(ids[0]))
    return [{'feedback_index':i,'label':int(y[i]),'image_28x28_ascii':ascii_image(x[i])} for i in chosen[:n]]


def spec(task,state):
    labels=list(range(10)) if task=='mnist' else ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    arch=('x.flatten(1); h=relu(linear(x,fc1.weight,fc1.bias)); logits=linear(h,fc2.weight,fc2.bias)' if task=='mnist' else
          'h=max_pool2d(relu(conv2d(x,conv1.weight,conv1.bias,padding=1)),2); h=max_pool2d(relu(conv2d(h,conv2.weight,conv2.bias,padding=1)),2); logits=linear(h.flatten(1),head.weight,head.bias)')
    return {'task':task,'objective':'Maximize held-out ten-class classification accuracy',
            'input':'float32 [N,1,28,28], pixel/255 in [0,1], black background 0, rows top to bottom, columns left to right',
            'class_indices':labels,'forward':arch,'tensor_shapes':{k:list(v.shape) for k,v in state.items()},
            'conventions':'PyTorch linear weights [out,in]; convolution [out,in,kernel_row,kernel_col], cross-correlation stride1; maxpool stride2, no padding; flatten channel then row then column; output argmax',
            'parameter_count':sum(v.size for v in state.values())}


class Budget:
    def __init__(self):
        self.dir=ROOT/'runtime';self.dir.mkdir(exist_ok=True)
        self.path=self.dir/'usage.json';self.lock=self.dir/'usage.lock'
    def reserve(self,label,chars,maxout):
        # Conservative upper bound: input tokens <= UTF-8 bytes, plus template allowance.
        inp,out=PRICES[MODEL]
        bound=(chars+4096)*inp/1e6+maxout*out/1e6
        with self.lock.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            d=json.loads(self.path.read_text())
            if d['blocked'] or len(d['calls'])>=d['max_calls'] or d['spent']+d['reserved']+bound>d['cap']:
                d['blocked']=True;d['blocked_reason']='API request/spend limit or existing billing block'
                dump(self.path,d);raise RuntimeError('API budget stop')
            i=len(d['calls']);d['calls'].append({'label':label,'model':MODEL,'prices':[inp,out],'reserved':bound,'status':'pending','time':time.time()});d['reserved']+=bound
            dump(self.path,d);return i
    def finish(self,i,usage,error=None):
        with self.lock.open('a+') as f:
            fcntl.flock(f,fcntl.LOCK_EX);d=json.loads(self.path.read_text());entry=d['calls'][i]
            if usage is None:
                # Unknown billing: keep reservation and block future dispatch.
                entry.update(status='unknown_billing',error=error);d['blocked']=True
            else:
                inp,out=entry.get('prices',[.15,.60])
                cost=usage['prompt_tokens']*inp/1e6+usage['completion_tokens']*out/1e6
                d['reserved']-=entry['reserved'];d['spent']+=cost
                entry.update(status='complete',usage=usage,cost=cost)
            dump(self.path,d)


def parse_response(r):
    choice=r['choices'][0];text=choice['message'].get('content') or ''
    if choice.get('finish_reason')!='stop':raise ValueError('Incomplete LLM response: '+str(choice.get('finish_reason')))
    blocks=re.findall(r'```(?:python|py)\s*\n(.*?)```',text,re.S)
    if len(blocks)!=1:raise ValueError('Expected exactly one Python block')
    source=blocks[0].strip()+'\n';ast.parse(source)
    return source,text


def evaluate_state(task,state,x,y,device,features=False):
    net=Network(task).to(device);net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in state.items()});net.eval()
    probs=[];feats=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):
            logits,h=net(torch.from_numpy(x[start:start+256]).to(device),trace=True)
            probs.append(logits.softmax(-1).cpu().numpy())
            if features:feats.append(h.cpu().numpy())
    p=np.concatenate(probs);pred=p.argmax(-1);correct=pred==y
    truep=p[np.arange(len(y)),y];ce=-np.log(np.maximum(truep,1e-30))
    # Accuracy dominates; probability supplies a small tie-sensitive signal.
    scores=.9*correct+.1*truep
    return {'accuracy':float(correct.mean()),'cross_entropy':float(ce.mean()),'score':float(scores.mean()),
            'probs':p,'pred':pred,'scores':scores,'features':np.concatenate(feats) if features else None}


def summary(result):return {k:result[k] for k in ['accuracy','cross_entropy','score']}


def mutation_api_calls(folder):
    # Count dispatches including provider failures. Cached replays are not calls.
    ids=set()
    for directory in (folder/'llm').glob('mutation_*'):
        for name in ['meta.json','dispatch.json']:
            for path in directory.rglob(name):
                record=json.loads(path.read_text())
                if 'ledger_call' in record:ids.add(record['ledger_call'])
    return len(ids)


def weight_summary(state):
    result={}
    for k,v in state.items():
        item={'shape':list(v.shape),'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max()),'nonzero':int(np.count_nonzero(v)),'first_values':v.ravel()[:24].round(4).tolist()}
        if v.ndim==4:item['first_filters']=v[:min(4,len(v))].round(3).tolist()
        if v.ndim==1:item['values']=v.round(4).tolist()
        result[k]=item
    return result


class Adapter:
    def __init__(self,task,seed,device,folder,compiler,x,y):
        self.task=task;self.seed=seed;self.device=device;self.folder=folder;self.compiler=compiler;self.x=x;self.y=y
        self.proposals=0;self.recent=[];self.source_history={};self.eval_calls=0

    def get_adapter_state(self):
        return {'proposals':self.proposals,'recent':list(self.recent),'eval_calls':self.eval_calls}

    def set_adapter_state(self,state):
        self.proposals=state.get('proposals',0);self.recent=list(state.get('recent',[]));self.eval_calls=state.get('eval_calls',0)

    def batch_evaluate(self,items):
        # GEPA's default batch helper requests traces even for full validation.
        # Only feedback batches may produce trajectories for reflection.
        return [self.evaluate(batch,candidate,capture_traces=batch[0]['split']=='feedback')
                for candidate,batch in items]


def run_adam(task,seed,device,folder,epochs=50):
    folder.mkdir(parents=True,exist_ok=True)
    fb=load_npz(ROOT/'data'/task/f'seed{seed}'/'feedback.npz');sv=load_npz(ROOT/'data'/task/f'seed{seed}'/'selection.npz')
    xx=torch.from_numpy(fb['x']).to(device);yy=torch.from_numpy(fb['y']).to(device)
    best=(-1.,float('inf'));beststate=None;history=[];start=time.time();steps=0
    for lr in [1e-4,1e-3,1e-2]:
        net=Network(task).to(device);net.load_state_dict({k:torch.from_numpy(v) for k,v in initial(task,seed).items()})
        opt=torch.optim.Adam(net.parameters(),lr=lr);rng=torch.Generator().manual_seed(20000+seed)
        stale=0;localbest=(-1.,float('inf'))
        for epoch in range(epochs):
            net.train();order=torch.randperm(len(yy),generator=rng).to(device)
            for ids in order.split(128):
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(net(xx[ids]),yy[ids]);loss.backward();opt.step();steps+=1
            state={k:v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
            r=evaluate_state(task,state,sv['x'],sv['y'],device);key=(r['score'],-r['cross_entropy'])
            history.append({'lr':lr,'epoch':epoch+1,**summary(r)})
            if key>(best[0],-best[1]):
                best=(r['score'],r['cross_entropy']);beststate=state;bestinfo={'lr':lr,'epoch':epoch+1,**summary(r)}
            if key>(localbest[0],-localbest[1]):localbest=(r['score'],r['cross_entropy']);stale=0
            else:stale+=1
            if stale>=10:break
    np.savez(folder/'adam_best.npz',**beststate)
    report={'task':task,'seed':seed,'best':bestinfo,'steps':steps,'epochs_limit':epochs,'wall_seconds':time.time()-start,'history':history}
    dump(folder/'adam_result.json',report)
    print(json.dumps({'stage':'adam_complete','task':task,'seed':seed,'best':bestinfo,'wall_seconds':time.time()-start}),flush=True)
    return report
