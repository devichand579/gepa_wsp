"""GEPA over bounded declarative parameter edits; no generated Python executes."""
from __future__ import annotations
import argparse,hashlib,json,os,re,subprocess,tempfile,time,urllib.request
from pathlib import Path
import numpy as np
import torch
import gepa
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
import experiment as common
from tensor_dsl import SPEC

ROOT=common.ROOT
SYSTEM=('You design the numerical parameters of a fixed neural network. Return one JSON object containing '
        'hypothesis, prediction, and ops in the specified parameter-edit language. Choose all coefficients '
        'using your reasoning and the supplied task/feedback. No Python programs, learners, fitting algorithms, '
        'network or layer execution, data access, optimizers, gradients, or candidate search execute inside edits. '
        'Only the listed parameter operations execute. All network inference is performed by the harness. '
        'No subsequent training occurs: the resulting tensors alone must perform the task.')

def language_spec(task):
    module=common.modules(task)[0];weight=module+'.weight';bias=module+'.bias'
    ix=[0,':'] if task=='mnist' else [0,0,':',':']
    dest=[1,':'] if task=='mnist' else [1,0,':',':']
    shape=[28,28] if task=='mnist' else [3,3]
    examples=[{'op':'set','tensor':bias,'value':0},
              {'op':'set','tensor':weight,'index':ix,'reshape':shape,'expr':'0.01*(x0-x1)'},
              {'op':'copy','tensor':weight,'index':dest,'source_tensor':weight,'source_index':ix},
              {'op':'normalize','tensor':weight,'axes':1 if task=='mnist' else [1,2,3],'norm':1}]
    examples.append({'op':'set','tensor':'fc2.weight' if task=='mnist' else 'head.weight',
                     'index':[0,':'],'reshape':[128] if task=='mnist' else [32,7,7],
                     'expr':'0.01*x0' if task=='mnist' else '0.01*(x1+x2)'})
    return SPEC.split('Examples of syntax only:')[0]+'Prefer compact formulas and block operations over long lists of numbers. Literal value arrays should contain at most 64 numbers. A simple approximate design is useful; you need not solve the entire task in one step. Unused parameters may be set to zero. Operations execute in order. Invalid operations are rejected individually without changing weights; other valid operations still execute. Rejections are reported, never repaired. A submission with no valid operations fails.\nExamples of valid syntax for this architecture only (not a recommended classification strategy):\n'+json.dumps(examples)

def output_schema(tensors):
    number={'type':'number'};integer={'type':'integer'}
    bound={'anyOf':[integer,{'type':'null'}]}
    indexes={'type':'array','maxItems':4,'items':{'anyOf':[integer,{'type':'string','enum':[':']},
        {'type':'array','minItems':2,'maxItems':3,'items':bound}]}}
    shape={'type':'array','minItems':1,'maxItems':4,'items':integer}
    props={'op':{'type':'string','enum':['set','add','scale','scatter','copy','normalize']},
           'tensor':{'type':'string','enum':tensors},'index':indexes,'reshape':shape,
           'value':{'$ref':'#/$defs/numeric_value'},'expr':{'type':'string','maxLength':4000},
           'source_tensor':{'type':'string','enum':tensors},'source_index':indexes,'source_reshape':shape,
           'transpose':shape,'shift':shape,'factor':number,'offset':number,
           'axes':{'anyOf':[integer,shape]},'center':{'type':'boolean'},'norm':number,
           'coordinates':{'type':'array','maxItems':4096,'items':shape}}
    return {'type':'object','properties':{'hypothesis':{'type':'string','maxLength':1500},
            'prediction':{'type':'string','maxLength':1000},
            'ops':{'type':'array','minItems':1,'maxItems':256,'items':{
                'type':'object','properties':props,'required':['op','tensor'],'additionalProperties':False}}},
            'required':['hypothesis','prediction','ops'],'additionalProperties':False,
            '$defs':{'numeric_value':{'anyOf':[number,
                {'type':'array','maxItems':64,'items':number},
                {'type':'array','maxItems':8,'items':{'type':'array','maxItems':8,'items':number}}]}}}

def parse_response(response):
    choice=response['choices'][0]
    if choice.get('finish_reason') not in ['stop','tool_calls']:raise ValueError('Incomplete response: '+str(choice.get('finish_reason')))
    tool_calls=choice['message'].get('tool_calls')
    if tool_calls:
        assert len(tool_calls)==1 and tool_calls[0]['function']['name']=='submit_tensor_edits'
        arguments=tool_calls[0]['function']['arguments']
        program=json.loads(arguments) if isinstance(arguments,str) else arguments
        assert isinstance(program,dict) and isinstance(program.get('ops'),list)
        return program
    content=choice['message'].get('content') or ''
    text=content.strip()
    if text.startswith('```'):
        assert text.endswith('```');text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    program=json.loads(text)
    assert isinstance(program,dict) and isinstance(program.get('ops'),list)
    return program

def call(prompt,folder,label):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'service_failure.json').exists():
        raise RuntimeError('Previously dispatched provider failure; counts as a proposal and is not retried')
    key=Path('/tmp/gepa-weight-space-fireworks.key').read_text().strip()
    assert key not in prompt and 'selection_index' not in prompt and 'test_index' not in prompt
    payload={'model':common.MODEL,'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],
             'temperature':.6,'reasoning_effort':common.EFFORT,'max_tokens':int(os.environ.get('STRICT_MAX_OUTPUT','32768'))}
    if os.environ.get('STRICT_THINKING_BUDGET'):
        payload.pop('reasoning_effort')
        payload['thinking']={'type':'enabled','budget_tokens':int(os.environ['STRICT_THINKING_BUDGET'])}
    if os.environ.get('STRICT_STRUCTURED','0')=='1':
        start=prompt.index('{"network":');info=json.JSONDecoder().raw_decode(prompt[start:])[0]
        payload['response_format']={'type':'json_schema','json_schema':{'name':'TensorEdits',
            'schema':output_schema(list(info['network']['tensor_shapes']))}}
    if os.environ.get('STRICT_TOOL_MODE','0')=='1':
        payload.pop('response_format',None)
        start=prompt.index('{"network":');info=json.JSONDecoder().raw_decode(prompt[start:])[0]
        payload['tools']=[{'type':'function','function':{'name':'submit_tensor_edits',
            'description':'Submit exactly one bounded neural-parameter edit attempt. No Python, network evaluation or fitting executes. Choose the numerical coefficients and compact parameter operations now.',
            'parameters':output_schema(list(info['network']['tensor_shapes']))}}]
        payload['tool_choice']={'type':'function','function':{'name':'submit_tensor_edits'}}
    if (folder/'response.json').exists():
        if json.loads((folder/'request.json').read_text())==payload:
            common.append(folder/'replays.jsonl',{'time':time.time(),'exact_request_replay':True})
            return parse_response(json.loads((folder/'response.json').read_text()))
        previous=folder;n=1
        while (previous/f'restart_{n:03d}').exists():n+=1
        folder=previous/f'restart_{n:03d}';folder.mkdir()
    common.dump(folder/'request.json',payload)
    budget=common.Budget();call_id=budget.reserve(label,len(json.dumps(payload['messages']).encode()),payload['max_tokens'])
    common.dump(folder/'dispatch.json',{'ledger_call':call_id,'label':label,'time':time.time()})
    req=urllib.request.Request('https://api.fireworks.ai/inference/v1/chat/completions',data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.time()
    try:
        with urllib.request.urlopen(req,timeout=600) as f:response=json.load(f)
    except Exception as e:
        common.dump(folder/'service_failure.json',{'ledger_call':call_id,'error_type':type(e).__name__,
                                                  'http_status':getattr(e,'code',None)})
        budget.finish(call_id,None,type(e).__name__);raise RuntimeError('Provider failure: '+type(e).__name__) from None
    usage=response.get('usage')
    if not usage or not all(k in usage for k in ['prompt_tokens','completion_tokens']):
        common.dump(folder/'service_failure.json',{'ledger_call':call_id,'error_type':'missing_usage'})
        budget.finish(call_id,None,'missing_usage');raise RuntimeError('Missing billing usage')
    budget.finish(call_id,usage)
    assert key not in json.dumps(response)
    common.dump(folder/'response.json',response)
    (folder/'answer.md').write_text(response['choices'][0]['message'].get('content') or '')
    common.dump(folder/'meta.json',{'elapsed':time.time()-start,'usage':usage,'ledger_call':call_id,'finish_reason':response['choices'][0].get('finish_reason')})
    return parse_response(response)

def execute(state,program,allowed,return_status=False):
    numpy_root=Path(np.__file__).parent
    with tempfile.TemporaryDirectory(prefix='gepa-declarative-') as name:
        folder=Path(name);np.savez(folder/'input.npz',**state)
        (folder/'program.json').write_text(json.dumps(program));(folder/'allowed.json').write_text(json.dumps(allowed))
        command=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv',
            '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
            '--ro-bind',str(numpy_root),'/opt/numpy','--ro-bind',str(numpy_root.parent/'numpy.libs'),'/opt/numpy.libs',
            '--ro-bind',str(ROOT/'tensor_dsl.py'),'/opt/tensor_dsl.py',
            '--ro-bind',str(ROOT/'tensor_dsl_worker.py'),'/worker.py','--bind',str(folder),'/work',
            '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/work',
            '--setenv','PYTHONPATH','/opt','--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
            '/usr/bin/python3','-s','/worker.py']
        p=subprocess.run(command,capture_output=True,text=True,timeout=35,env={'PATH':'/usr/bin:/bin'})
        if p.returncode:
            details=(folder/'status.json').read_text() if (folder/'status.json').exists() else p.stderr[-3500:]
            raise ValueError('Declarative edit rejected: '+details[:12000])
        result=common.load_npz(folder/'output.npz')
        status=json.loads((folder/'status.json').read_text())
    # Trusted host independently checks outputs; no generated code can alter this.
    assert set(result)==set(state)
    for k,v in result.items():
        assert v.shape==state[k].shape and v.dtype==np.float32 and np.isfinite(v).all()
        assert np.max(np.abs(v),initial=0)<=1000
        if k not in allowed:assert np.array_equal(v,state[k])
    return (result,status) if return_status else result

class Compiler:
    def __init__(self,task,state,folder):
        self.task=task;self.seed=state;self.folder=folder;folder.mkdir(exist_ok=True,parents=True)
        np.savez(folder/'base.npz',**state)
        sha=hashlib.sha256()
        for k,v in sorted(state.items()):sha.update(k.encode());sha.update(str(v.shape).encode());sha.update(v.tobytes())
        self.seed_hash=sha.hexdigest()
    def key(self,candidate):return common.digest({'base_weights':self.seed_hash,'strict_programs':candidate})
    def materialize(self,candidate):
        h=self.key(candidate);path=self.folder/f'{h}.npz'
        if path.exists():return h,common.load_npz(path)
        state={k:v.copy() for k,v in self.seed.items()}
        diagnostics={}
        for module in common.modules(self.task):
            item=json.loads(candidate[module]);parent=item['parent'];assert re.fullmatch('[a-f0-9]{64}|base',parent)
            allowed=[k for k in state if k.startswith(module+'.')]
            updated=common.load_npz(self.folder/f'{parent}.npz')
            if item['program']['ops']:
                updated,status=execute(updated,item['program'],allowed,return_status=True)
                diagnostics[module]=status
            elif (self.folder/f'{parent}.joint_status.json').exists():
                diagnostics[module]=json.loads((self.folder/f'{parent}.joint_status.json').read_text())
            for k in allowed:state[k]=updated[k]
        np.savez(path,**state);common.dump(self.folder/f'{h}.json',candidate)
        common.dump(self.folder/f'{h}.diagnostics.json',diagnostics)
        return h,state

class Adapter(common.Adapter):
    def evaluate(self,batch,candidate,capture_traces=False):
        result=super().evaluate(batch,candidate,capture_traces)
        if batch[0]['split']=='feedback':
            h=self.compiler.key(candidate);path=self.compiler.folder/f'{h}.diagnostics.json'
            if path.exists():
                for record in self.recent:
                    if record.get('child')==h:record['operation_diagnostics']=json.loads(path.read_text())
        return result
    def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
        self.proposals+=1;module=components_to_update[0];parent,parent_state=self.compiler.materialize(candidate)
        referenced={}
        for entry in candidate.values():
            pid=json.loads(entry)['parent'];path=self.compiler.folder/f'{pid}.joint_program.json'
            if path.exists():referenced[pid]=json.loads(path.read_text())
        payload={'network':common.spec(self.task,self.compiler.seed),'selected_modules':components_to_update,
                 'current_candidate_programs':candidate,
                 'referenced_joint_programs':referenced,
                 'construction_program_attempt':json.loads((self.folder/'construction.json').read_text()),
                 'construction_status':json.loads((self.folder/'construction_status.json').read_text()),
                 'current_operation_diagnostics':json.loads((self.compiler.folder/f'{parent}.diagnostics.json').read_text()),
                 'feedback':reflective_dataset[module],'recent_proposals':self.recent[-4:]}
        prompt=language_spec(self.task)+'\n\nMutate ONLY tensors within these modules: '+json.dumps(components_to_update)+'. All other tensors must remain unchanged. '
        prompt+='The parameter state is the full parent checkpoint. A failed construction was NOT applied: in that case the starting checkpoint was the random initialization. The attempted construction and its error are supplied for diagnosis only. Diagnose from supplied feedback, choose ONE coherent parameter update, and predict its effect. '
        prompt+='Return the complete JSON object, not a plan or Python code.\n'+json.dumps(payload)
        pf=self.folder/'llm'/f'mutation_{self.proposals:03d}'
        try:
            program=call(prompt,pf,f'strict/{self.task}/seed{self.seed}/mutation{self.proposals}')
            assert 1<=len(program['ops'])<=256
            if len(components_to_update)>1:
                allowed=[k for k in parent_state if k.split('.')[0] in components_to_update]
                updated,status=execute(parent_state,program,allowed,return_status=True)
                joint=common.digest({'parent':parent,'joint_program':program})
                np.savez(self.compiler.folder/f'{joint}.npz',**updated)
                common.dump(self.compiler.folder/f'{joint}.joint_program.json',program)
                common.dump(self.compiler.folder/f'{joint}.joint_status.json',status)
                entry=json.dumps({'parent':joint,'program':{'ops':[]}})
            else:entry=json.dumps({'parent':parent,'program':program})
            updates={m:entry for m in components_to_update};child=dict(candidate);child.update(updates)
            self.recent.append({'modules':components_to_update,'parent':parent,'child':self.compiler.key(child),
                                'hypothesis':program.get('hypothesis','')[:1600],'prediction':program.get('prediction','')[:800]})
            common.dump(pf/'program.json',program)
            return updates
        except Exception as e:
            common.append(self.folder/'proposal_errors.jsonl',{'proposal':self.proposals,'error':str(e)})
            self.recent.append({'module':module,'error':str(e)})
            # Invalid operation is a failing candidate, not a free retry.
            return {m:json.dumps({'parent':parent,'program':{'ops':[{'op':'INVALID','tensor':m+'.weight'}]}}) for m in components_to_update}

def select_modules(state,trajectories,scores,candidate_idx,candidate):
    """Joint update, then each layer in order; schedule survives GEPA restarts."""
    names=list(candidate);slot=state.i%(len(names)+1)
    return names if slot==0 else [names[slot-1]]

def run(task,seed,device,folder,proposals):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'search_result.json').exists():return json.loads((folder/'search_result.json').read_text())
    fb=common.load_npz(ROOT/'data'/task/f'seed{seed}'/'feedback.npz')
    sv=common.load_npz(ROOT/'data'/task/f'seed{seed}'/'selection.npz')
    state=common.initial(task,seed);np.savez(folder/'initial.npz',**state)
    common.dump(folder/'run_config.json',{'task':task,'seed':seed,'device':device,'reflection_model':common.MODEL,
        'reasoning_effort':common.EFFORT,'temperature':.6,'max_output_tokens':int(os.environ.get('STRICT_MAX_OUTPUT','32768')),
        'thinking_budget':os.environ.get('STRICT_THINKING_BUDGET'),'structured_output':os.environ.get('STRICT_STRUCTURED','0'),
        'tool_mode':os.environ.get('STRICT_TOOL_MODE','0'),'module_schedule':'joint then each layer, cycling',
        'proposal_budget':proposals,
        'interface':'bounded declarative tensor edits','network':common.spec(task,state),
        'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['strict_experiment.py','tensor_dsl.py','tensor_dsl_worker.py','experiment.py']}})
    if (folder/'construction_status.json').exists() and (folder/'one_shot.npz').exists():
        constructed=common.load_npz(folder/'one_shot.npz');error=json.loads((folder/'construction_status.json').read_text())['error']
    else:
        prompt=language_spec(task)+'\n\nConstruct this untrained network in ONE submission with no evaluation feedback. You may edit ALL tensors. '
        prompt+='Choose the computations and coefficients yourself. Return one complete JSON object.\n'
        prompt+=json.dumps({'network':common.spec(task,state),'initial_weight_statistics':common.weight_summary(state),
                            'examples':common.example_packet(fb['x'],fb['y'])})
        program={'ops':[]};operation_status=None
        try:
            program=call(prompt,folder/'llm'/'construction',f'strict/{task}/seed{seed}/construction')
            constructed,operation_status=execute(state,program,list(state),return_status=True);error=None
        except Exception as e:constructed=state;error=str(e)
        common.dump(folder/'construction.json',program);np.savez(folder/'one_shot.npz',**constructed)
        common.dump(folder/'construction_status.json',{'error':error,'operations':operation_status,
            'fully_valid':error is None and not operation_status['rejected']})
    assert not json.loads((ROOT/'runtime/usage.json').read_text())['blocked'],'Service/billing stop'
    one=common.evaluate_state(task,constructed,sv['x'],sv['y'],device)
    common.dump(folder/'one_shot_selection.json',common.summary(one))
    print(json.dumps({'stage':'strict_construction','task':task,'seed':seed,'selection':common.summary(one),'error':error}),flush=True)
    compiler=Compiler(task,constructed,folder/'checkpoints')
    candidate={m:json.dumps({'parent':'base','program':{'ops':[]}}) for m in common.modules(task)}
    adapter=Adapter(task,seed,device,folder,compiler,{'feedback':fb['x'],'selection':sv['x']},{'feedback':fb['y'],'selection':sv['y']})
    result=gepa.optimize(seed_candidate=candidate,
        trainset=[{'split':'feedback','index':i} for i in range(len(fb['y']))],
        valset=[{'split':'selection','index':i} for i in range(len(sv['y']))],adapter=adapter,
        candidate_selection_strategy='pareto',skip_perfect_score=False,reflection_minibatch_size=128,
        module_selector=select_modules,use_merge=False,
        stop_callbacks=[MaxCandidateProposalsStopper(proposals+20),lambda state:common.mutation_api_calls(folder)>=proposals,
                        lambda state:json.loads((ROOT/'runtime/usage.json').read_text())['blocked']],
        max_metric_calls=1000+(proposals+20)*1256+100,run_dir=str(folder/'gepa'),seed=seed,track_best_outputs=False,
        raise_on_exception=True)
    assert not json.loads((ROOT/'runtime/usage.json').read_text())['blocked'],'Service/billing stop'
    h,best=compiler.materialize(result.best_candidate);np.savez(folder/'gepa_best.npz',**best)
    final=common.evaluate_state(task,best,sv['x'],sv['y'],device)
    report={'task':task,'seed':seed,'one_shot':common.summary(one),'gepa':common.summary(final),
            'best_candidate':h,'proposals':common.mutation_api_calls(folder),'gepa_iterations':adapter.proposals,
            'tracked_candidates':result.num_candidates,'metric_calls':result.total_metric_calls,
            'val_scores':result.val_aggregate_scores,'parents':result.parents,'construction_error':error,
            'changed_weights':{k:int(np.count_nonzero(best[k]!=constructed[k])) for k in best},
            'interface':'bounded declarative tensor edits; no generated Python'}
    common.dump(folder/'search_result.json',report);common.dump(folder/'gepa_result.json',result.to_dict())
    print(json.dumps({'stage':'strict_search_complete',**report}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['mnist','fashion']);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda:0');p.add_argument('--proposals',type=int,default=1);p.add_argument('--phase',choices=['strict_pilot','main'],default='strict_pilot')
    a=p.parse_args();torch.set_num_threads(2);run(a.task,a.seed,a.device,ROOT/'runs'/a.phase/a.task/f'seed{a.seed}',a.proposals)
