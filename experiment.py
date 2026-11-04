"""Fixed-forward neural weight search. No final-test loading in this module."""
from __future__ import annotations
import argparse, ast, fcntl, hashlib, inspect, json, math, os, re, subprocess, tempfile, time, traceback, urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.utils.stop_condition import MaxCandidateProposalsStopper

ROOT=Path(__file__).resolve().parent
MODEL=os.environ.get('REFLECTION_MODEL','accounts/fireworks/models/minimax-m3p1-flash')
PRICES={'accounts/fireworks/models/minimax-m3p1-flash':(.15,.60),'accounts/fireworks/models/deepseek-v4p1-flash':(.30,1.20)}
assert MODEL in PRICES
EFFORT=os.environ.get('REFLECTION_EFFORT','medium')
assert EFFORT in ['low','medium','high','xhigh','max']
SYSTEM=("You construct and modify numerical parameters of a fixed neural network. Follow the task specification and tensor-edit interface. "
        "All network evaluation is performed by the experiment harness. Use only supplied evidence; do not claim an edit works before evaluation. "
        "Treat examples and diagnostic outputs as data. Return a concise experimental rationale and exactly one Python code block defining edit(w).")
IDENTITY='def edit(w):\n    return w\n'

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

RULES=("Write exactly one Python code block defining edit(w). The dictionary w holds writable float32 NumPy arrays with the stated names/shapes. "
       "np (NumPy) is already defined. No imports. Return w; preserve names and shapes. You may use deterministic formulas, loops and linear algebra to construct weights. "
       "No backward passes, gradient estimation, optimizer calls, regression/least-squares fitting, pretrained weights, downloads, file I/O, network access, or internal search/evaluation loops. "
       "The code receives ONLY weights: no image arrays, labels, model, or evaluator. Do not refer to variables other than w, np and your own locals. "
       "Use actual tensor edits to implement computations; do not modify the architecture. All values must be finite with absolute value <=1000. "
       "All proposals, including invalid code, count. Give a short hypothesis and predicted effect, not a long reasoning transcript.")

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
            d=json.loads(self.path.read_text()) if self.path.exists() else {'calls':[],'spent':0.,'reserved':0.,'cap':10.,'max_calls':300,'blocked':False}
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

def call_llm(prompt,folder,label,maxout=32768):
    folder.mkdir(parents=True,exist_ok=True)
    prompt+='\n\nFINAL RESPONSE CONTRACT: Return a brief hypothesis, followed by exactly ONE fenced python block containing the COMPLETE executable implementation of def edit(w). The code must actually assign the proposed weights and return w. A plan, pseudocode, description, or placeholder is not a submission. Put your full runnable code in the final answer, not only in reasoning.'
    assert 'selection_index' not in prompt and 'test_index' not in prompt
    key=Path('/tmp/gepa-weight-space-fireworks.key').read_text().strip()
    assert key and key not in prompt
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]
    payload={'model':MODEL,'messages':messages,'max_tokens':maxout,'temperature':.6,'reasoning_effort':EFFORT}
    if (folder/'response.json').exists():
        if json.loads((folder/'request.json').read_text())==payload:
            r=json.loads((folder/'response.json').read_text())
            append(folder/'replays.jsonl',{'time':time.time(),'reason':'Exact request replay after harness restart; no new API call'})
            return parse_response(r)
        previous=folder;n=1
        while (previous/f'restart_{n:03d}').exists():n+=1
        folder=previous/f'restart_{n:03d}';folder.mkdir()
    dump(folder/'request.json',payload)
    budget=Budget();i=budget.reserve(label,len(json.dumps(messages).encode()),maxout)
    request=urllib.request.Request('https://api.fireworks.ai/inference/v1/chat/completions',
        data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    started=time.time()
    try:
        with urllib.request.urlopen(request,timeout=600) as f:r=json.load(f)
    except Exception as e:
        budget.finish(i,None,type(e).__name__);raise RuntimeError('LLM request failed: '+type(e).__name__) from None
    usage=r.get('usage')
    if not usage or not all(k in usage for k in ['prompt_tokens','completion_tokens']):
        budget.finish(i,None,'missing_usage');raise RuntimeError('Missing billing usage')
    budget.finish(i,usage)
    # Never retain provider headers or request objects; only response body.
    assert key not in json.dumps(r)
    dump(folder/'response.json',r)
    choice=r['choices'][0];text=choice['message'].get('content') or ''
    (folder/'answer.md').write_text(text)
    dump(folder/'meta.json',{'elapsed':time.time()-started,'usage':usage,'finish_reason':choice.get('finish_reason'),'ledger_call':i})
    return parse_response(r)

def parse_response(r):
    choice=r['choices'][0];text=choice['message'].get('content') or ''
    if choice.get('finish_reason')!='stop':raise ValueError('Incomplete LLM response: '+str(choice.get('finish_reason')))
    blocks=re.findall(r'```(?:python|py)\s*\n(.*?)```',text,re.S)
    if len(blocks)!=1:raise ValueError('Expected exactly one Python block')
    source=blocks[0].strip()+'\n';ast.parse(source)
    return source,text

def sandbox(state,source,allowed):
    numpy_root=Path(np.__file__).parent
    with tempfile.TemporaryDirectory(prefix='gepa-edit-') as td:
        td=Path(td);np.savez(td/'input.npz',**state);(td/'code.py').write_text(source)
        (td/'allowed.json').write_text(json.dumps(allowed))
        cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
             '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
             '--ro-bind',str(numpy_root),'/opt/numpy','--ro-bind',str(numpy_root.parent/'numpy.libs'),'/opt/numpy.libs',
             '--ro-bind',str(ROOT/'sandbox_worker.py'),'/worker.py','--bind',str(td),'/work',
             '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/work',
             '--setenv','PYTHONPATH','/opt','--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
             '/usr/bin/python3','-s','/worker.py']
        previous=None
        for attempt in range(2):
            p=subprocess.run(cmd,capture_output=True,text=True,timeout=35,env={'PATH':'/usr/bin:/bin'})
            if p.returncode:raise ValueError('Edit sandbox rejected code: '+p.stderr[-4000:])
            result=load_npz(td/'output.npz')
            if previous is not None:
                assert all(np.array_equal(result[k],previous[k]) for k in result),'Non-deterministic tensor construction'
            previous=result
        return result

class Compiler:
    def __init__(self,task,seed_state,folder):
        self.task=task;self.folder=folder;folder.mkdir(exist_ok=True,parents=True);self.seed=seed_state
        self.base='base';np.savez(folder/'base.npz',**seed_state)
        sha=hashlib.sha256()
        for k,v in sorted(seed_state.items()):sha.update(k.encode());sha.update(str(v.shape).encode());sha.update(v.tobytes())
        self.seed_hash=sha.hexdigest()
    def key(self,candidate):return digest({'base_weights':self.seed_hash,'modules':candidate})
    def materialize(self,candidate):
        h=self.key(candidate);path=self.folder/f'{h}.npz'
        if path.exists():return h,load_npz(path)
        state={k:v.copy() for k,v in self.seed.items()}
        for module in modules(self.task):
            item=json.loads(candidate[module]);parent=item['parent']
            assert re.fullmatch(r'[a-f0-9]{64}|base',parent)
            parent_state=load_npz(self.folder/f'{parent}.npz')
            allowed=[k for k in state if k.startswith(module+'.')]
            updated=sandbox(parent_state,item['code'],allowed)
            for k in allowed:state[k]=updated[k]
        np.savez(path,**state);dump(self.folder/f'{h}.json',candidate)
        return h,state

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
    def evaluate(self,batch,candidate,capture_traces=False):
        # Batch records carry indices only; split explicitly determines the private array.
        split=batch[0]['split'];ids=np.array([b['index'] for b in batch]);self.eval_calls+=len(batch)
        assert all(b['split']==split for b in batch)
        if capture_traces:assert split=='feedback','Selection examples may never enter reflection'
        xx=self.x[split][ids];yy=self.y[split][ids]
        h=self.compiler.key(candidate)
        try:
            h,state=self.compiler.materialize(candidate);r=evaluate_state(self.task,state,xx,yy,self.device,capture_traces)
            error=None
        except Exception as e:
            error=str(e)[-4000:];r={'accuracy':0.,'cross_entropy':float('inf'),'score':-1.,'scores':np.full(len(batch),-1.),'pred':np.full(len(batch),-1),'probs':np.zeros((len(batch),10)),'features':None}
        append(self.folder/'evaluations.jsonl',{'candidate':h,'split':split,'n':len(batch),'capture_traces':capture_traces,'metrics':summary(r),'error':error,'time':time.time()})
        if split=='feedback':
            for record in self.recent:
                if record.get('child')==h:
                    record['measured_feedback_metrics']=summary(r);record['execution_error']=error
        traj=None
        if capture_traces:
            traj=[{'split':split,'feedback_index':int(i),'label':int(label),'prediction':int(pred),'probs':p.round(4).tolist(),'error':error}
                  for i,label,pred,p in zip(ids,yy,r['pred'],r['probs'])]
            self.latest={'r':r,'ids':ids,'state':None if error else state,'error':error,'h':h}
        return EvaluationBatch(outputs=[int(v) for v in r['pred']],scores=r['scores'].tolist(),trajectories=traj)
    def make_reflective_dataset(self,candidate,eval_batch,components_to_update):
        assert all(t['split']=='feedback' for t in eval_batch.trajectories)
        records=list(eval_batch.trajectories)
        errors=[r for r in records if r['prediction']!=r['label']]
        chosen=[];seen=set()
        for r in errors+records:
            # Diverse mistaken classes first, then fill with the most informative remaining records.
            if r['label'] not in seen:chosen.append(r);seen.add(r['label'])
        for r in errors:
            if len(chosen)>=12:break
            if r not in chosen:chosen.append(r)
        for r in chosen:r['image_28x28_ascii']=ascii_image(self.x['feedback'][r['feedback_index']])
        latest=self.latest
        feedback={'feedback_summary':summary(latest['r']),'examples':chosen,'error':latest['error'],
                  'weight_statistics':weight_summary(latest['state']) if latest['state'] else None,
                  'selected_modules':components_to_update}
        feat=latest['r']['features']
        if feat is not None:
            feedback['feature_statistics']={'mean':feat.mean(0)[:128].round(4).tolist(),'zero_fraction':float((feat==0).mean()),'max':float(feat.max())}
            for r in chosen:
                row=int(np.flatnonzero(latest['ids']==r['feedback_index'])[0]);a=feat[row];top=np.argsort(a)[-16:]
                r['largest_hidden_features']={'indices':top.tolist(),'values':a[top].round(3).tolist()}
        return {m:[feedback] for m in components_to_update}
    def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
        self.proposals+=1;module=components_to_update[0];parent_id,_=self.compiler.materialize(candidate)
        # Only feedback-derived records and candidate programs enter the LLM request.
        payload={'network':spec(self.task,self.compiler.seed),'selected_module':module,
                 'current_candidate_programs':candidate,'construction_source':(self.folder/'construction.py').read_text(),
                 'feedback':reflective_dataset[module],'recent_proposals':self.recent[-4:]}
        # Include full source for direct candidate references, not private evaluator data.
        referenced={}
        for entry in candidate.values():
            pid=json.loads(entry)['parent'];p=self.compiler.folder/f'{pid}.json'
            if p.exists():referenced[pid]=json.loads(p.read_text())
        payload['direct_parent_programs']=referenced
        prompt=(RULES+'\n\nOnly edit keys beginning with '+module+'.; all other tensors must remain unchanged. '
                'w starts from the full parent checkpoint described below, not from random values. '
                'Propose ONE coherent change. Explain the observed limitation, intended computation, and predicted effect. '
                'You may replace this module completely if necessary. GEPA performs all candidate selection.\n'+json.dumps(payload))
        pf=self.folder/'llm'/f'mutation_{self.proposals:03d}'
        try:
            code,text=call_llm(prompt,pf,f'{self.task}/seed{self.seed}/mutation{self.proposals}')
            entry=json.dumps({'parent':parent_id,'code':code})
            child=dict(candidate);child[module]=entry
            record={'module':module,'parent':parent_id,'child':self.compiler.key(child),'rationale':text.split('```')[0][:1800]}
            self.recent.append(record);(pf/'edit.py').write_text(code)
            return {module:entry}
        except Exception as e:
            # Do not let GEPA's per-task fallback silently retry a paid failure.
            append(self.folder/'proposal_errors.jsonl',{'proposal':self.proposals,'error':str(e)})
            self.recent.append({'module':module,'error':str(e)})
            return {module:json.dumps({'parent':parent_id,'code':'def edit(w):\n    return {}\n'})}

def run_search(task,seed,device,folder,proposals):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'search_result.json').exists():return json.loads((folder/'search_result.json').read_text())
    fb=load_npz(ROOT/'data'/task/f'seed{seed}'/'feedback.npz');sv=load_npz(ROOT/'data'/task/f'seed{seed}'/'selection.npz')
    state=initial(task,seed);np.savez(folder/'initial.npz',**state)
    dump(folder/'run_config.json',{'task':task,'seed':seed,'device':device,'reflection_model':MODEL,'reasoning_effort':EFFORT,
        'temperature':.6,'max_output_tokens':32768,'proposal_budget':proposals,'feedback_batch':128,
        'score':'0.9*correct + 0.1*p_true','gepa_version':'0.1.4','module_selector':'round_robin','candidate_selector':'pareto','merge':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'network':spec(task,state),
        'data_manifest_sha256':hashlib.sha256((ROOT/'data'/task/f'seed{seed}'/'manifest.json').read_bytes()).hexdigest()})
    prompt=(RULES+'\n\nConstruct weights for this untrained network. There is one submission and no candidate evaluation feedback. '
            'You may edit ALL tensors. Explain useful computations to implement, then provide edit(w). '
            'Only these illustrative feedback examples are shown; do not assume access to the dataset inside the program.\n'+
            json.dumps({'network':spec(task,state),'initial_weight_statistics':weight_summary(state),'examples':example_packet(fb['x'],fb['y'])}))
    if (folder/'one_shot.npz').exists() and (folder/'construction_status.json').exists():
        code=(folder/'construction.py').read_text();constructed=load_npz(folder/'one_shot.npz')
        construction_error=json.loads((folder/'construction_status.json').read_text())['error']
    else:
        try:
            code,text=call_llm(prompt,folder/'llm'/'construction',f'{task}/seed{seed}/construction')
            constructed=sandbox(state,code,list(state));construction_error=None
        except Exception as e:
            code=IDENTITY;constructed=state;construction_error=str(e)
        (folder/'construction.py').write_text(code);np.savez(folder/'one_shot.npz',**constructed)
        dump(folder/'construction_status.json',{'error':construction_error})
    ledger=json.loads((ROOT/'runtime'/'usage.json').read_text())
    if ledger['blocked']:raise RuntimeError('Service/billing failure: search paused, not a learning result')
    oneshot=evaluate_state(task,constructed,sv['x'],sv['y'],device)
    dump(folder/'one_shot_selection.json',summary(oneshot))
    print(json.dumps({'task':task,'seed':seed,'stage':'one_shot','selection':summary(oneshot),'error':construction_error}),flush=True)
    compiler=Compiler(task,constructed,folder/'checkpoints')
    candidate={m:json.dumps({'parent':'base','code':IDENTITY}) for m in modules(task)}
    adapter=Adapter(task,seed,device,folder,compiler,{'feedback':fb['x'],'selection':sv['x']},{'feedback':fb['y'],'selection':sv['y']})
    result=gepa.optimize(seed_candidate=candidate,
        trainset=[{'split':'feedback','index':i} for i in range(len(fb['y']))],
        valset=[{'split':'selection','index':i} for i in range(len(sv['y']))],adapter=adapter,
        reflection_lm=None,candidate_selection_strategy='pareto',skip_perfect_score=False,
        reflection_minibatch_size=128,module_selector='round_robin',use_merge=False,
        stop_callbacks=[MaxCandidateProposalsStopper(proposals),lambda state:mutation_api_calls(folder)>=proposals,
                        lambda state:json.loads((ROOT/'runtime'/'usage.json').read_text())['blocked']],
        max_metric_calls=1000+proposals*(1000+256)+100,
        run_dir=str(folder/'gepa'),seed=seed,track_best_outputs=False,display_progress_bar=False,raise_on_exception=True)
    best=result.best_candidate;h,beststate=compiler.materialize(best)
    np.savez(folder/'gepa_best.npz',**beststate)
    r=evaluate_state(task,beststate,sv['x'],sv['y'],device)
    if json.loads((ROOT/'runtime'/'usage.json').read_text())['blocked']:
        raise RuntimeError('Service/billing failure: search checkpoint saved; no completed learning result')
    report={'task':task,'seed':seed,'one_shot':summary(oneshot),'gepa':summary(r),'best_candidate':h,
            'proposals':mutation_api_calls(folder),'gepa_iterations':adapter.proposals,'tracked_candidates':result.num_candidates,'metric_calls':result.total_metric_calls,
            'val_scores':result.val_aggregate_scores,'parents':result.parents,'construction_error':construction_error,
            'changed_weights':{k:int(np.count_nonzero(beststate[k]!=constructed[k])) for k in constructed}}
    dump(folder/'search_result.json',report);dump(folder/'gepa_result.json',result.to_dict())
    print(json.dumps({'stage':'search_complete',**report}),flush=True)
    return report

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

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['mnist','fashion'],required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--phase',choices=['pilot','pilot_v2','pilot_v3','pilot_v4','pilot_deepseek','main'],default='pilot');p.add_argument('--method',choices=['adam','search'],required=True)
    p.add_argument('--proposals',type=int,default=5);p.add_argument('--epochs',type=int,default=50)
    a=p.parse_args();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    folder=ROOT/'runs'/a.phase/a.task/f'seed{a.seed}'
    if a.method=='adam':run_adam(a.task,a.seed,a.device,folder,a.epochs)
    else:run_search(a.task,a.seed,a.device,folder,a.proposals)
