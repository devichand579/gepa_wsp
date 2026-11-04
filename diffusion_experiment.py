"""Feedback-aware GEPA for a fixed convolutional epsilon-predicting DDPM."""
from __future__ import annotations
import argparse,hashlib,json,time,types
from pathlib import Path
import numpy as np
import torch
import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.utils.stop_condition import MaxCandidateProposalsStopper
import experiment as common
import rich_experiment as rich
import expansion_client as client
import diffusion_core as core
ROOT=common.ROOT
MODULES=['conv1','conv2','conv3','out']
BACKEND=types.SimpleNamespace(modules=lambda task:MODULES)
SYSTEM=rich.SYSTEM+' This task is unconditional diffusion epsilon prediction. Minimize mean squared error, not classification accuracy. Denoising improvement alone does not establish generation quality.'

def metrics(r):return {k:r[k] for k in ['mse','score']}
def load_data(seed,device):
 result={}
 for split,noise_seed in [('feedback',81000+seed),('selection',82000+seed)]:
  clean=common.load_npz(ROOT/'data/diffusion'/f'seed{seed}'/(split+'.npz'))['clean']
  x,t,y=core.fixed_batch(clean,noise_seed,device)
  result[split]={'clean':clean,'x':x.cpu().numpy(),'t':t.cpu().numpy(),'y':y.cpu().numpy()}
 return result

def forward(state,data,ids,device,capture=False,features=False):
 net=core.Denoiser().to(device);net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in state.items()});net.eval();parts={};handles=[];outputs=[];hidden=[]
 def hook(name):
  def record(module,args,out):
   parts.setdefault('layer.'+name+'.input0',[]).append(args[0].detach().cpu().numpy().copy())
   parts.setdefault('layer.'+name+'.output',[]).append(out.detach().cpu().numpy().copy())
  return record
 if capture:
  for name,module in net.named_modules():
   if name:handles.append(module.register_forward_hook(hook(name)))
 if features and not capture:
  handles.append(net.out.register_forward_pre_hook(lambda module,args:hidden.append(args[0].detach().cpu().numpy().copy())))
 try:
  with torch.inference_mode():
   for indices in np.array_split(np.asarray(ids,dtype=np.int64),max(1,int(np.ceil(len(ids)/128)))):
    x=torch.from_numpy(data['x'][indices]).to(device);t=torch.from_numpy(data['t'][indices]).to(device);outputs.append(net(x,t).cpu().numpy())
 finally:
  for h in handles:h.remove()
 prediction=np.concatenate(outputs);loss=((prediction-data['y'][ids])**2).reshape(len(ids),-1).mean(1)
 assert np.isfinite(prediction).all() and np.isfinite(loss).all(),'Nonfinite diffusion forward/loss'
 result={'mse':float(loss.mean()),'score':-float(loss.mean()),'scores':-loss,'prediction':prediction,'per_example_mse':loss}
 traces={k:np.concatenate(v) for k,v in parts.items()}
 if features:result['features']=traces['layer.out.input0'] if capture else np.concatenate(hidden)
 return result,traces

def render(image,scale=1):
 a=np.clip((np.asarray(image).reshape(16,16)/scale+1)/2,0,1);chars=' .:-=+*#%@'
 return '\n'.join(''.join(chars[min(9,int(v*10))] for v in row) for row in a)

def spec(state):
 return {'task':'MNIST16x16 unconditional DDPM epsilon prediction','parameter_count':sum(v.size for v in state.values()),
  'tensor_shapes':{k:list(v.shape) for k,v in state.items()},'input':'x_t float32[N,1,16,16]; t int64[N] in0..31; clean images in[-1,1]',
  'forward':'concat x_t with t/31 broadcast as second channel; conv1(2->32,3,pad1),ReLU; conv2(32->64,3,pad1),ReLU; conv3(64->64,3,pad1),ReLU; out(64->1,3,pad1), no output activation',
  'objective':'Minimize per-example pixel-mean squared error between epsilon prediction and sampled Gaussian epsilon. GEPA maximizes negative MSE.',
  'sampler':'32-step cosine-schedule DDPM; predicted x0 clipped[-1,1]; stochastic posterior with zero noise at final step',
  'conventions':'PyTorch convolution cross-correlation weights[out,in,kernel_row,kernel_col], stride1. All source arrays and coefficients are full precision in edit(w,f).'}

def feedback_snapshot(state,data,ids,device,seed,folder,parent,traces=None):
 all_ids=np.arange(len(data['x']));full,_=forward(state,data,all_ids,device,features=True)
 if traces is None:_,traces=forward(state,data,ids,device,capture=True)
 samples=core.sample(state,8,88000+seed,device)
 f={**{k:v for k,v in data.items()},'features':full['features'],'prediction':full['prediction'],'alphabar':core.schedule()['abar'].numpy(),
    'batch.ids':ids,'batch.x':data['x'][ids],'batch.t':data['t'][ids],'batch.y':data['y'][ids],'generated.samples':samples,**traces}
 chosen=list(ids[np.argsort(full['per_example_mse'][ids])[-6:]])
 for i in ids:
  if len(chosen)>=12:break
  if int(i) not in chosen:chosen.append(int(i))
 examples=[{'feedback_index':int(i),'timestep':int(data['t'][i]),'mse':float(full['per_example_mse'][i]),'clean_16x16':render(data['clean'][i]),'noisy_16x16_scaled_by3':render(data['x'][i],3),'predicted_epsilon_scaled_by3':render(full['prediction'][i],3)} for i in chosen]
 by_t={str(t):{'n':int((data['t']==t).sum()),'mse':float(full['per_example_mse'][data['t']==t].mean())} for t in np.unique(data['t'])}
 packet={'full_feedback_metrics':metrics(full),'timestep_metrics':by_t,'examples':examples,'current_weights':rich.weight_packet(state),
         'generated_feedback_samples':[render(x) for x in samples],
         'layer_statistics':{k:{'shape':list(v.shape),'mean':float(v.mean()),'std':float(v.std()),'zero_fraction':float((v==0).mean())} for k,v in traces.items()}}
 manifest={'split':'feedback','n_full_feedback':len(data['x']),'parent_checkpoint':parent,'batch_indices':ids.tolist(),'arrays_sha256':rich.arrays_digest(f),'array_schema':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in f.items()},'generated_feedback_samples':8,'reverse_steps':core.STEPS}
 common.append(folder/'forward_audit.jsonl',{'time':time.time(),'split':'feedback','purpose':'full-pool features plus cached batch traces and generated feedback samples','n':len(data['x']),'generated_forward_examples':8*core.STEPS,'parent':parent})
 return f,packet,manifest

def instructions(f):
 return ('Return one complete fenced python implementation of edit(w,f), optionally preceded by a brief hypothesis and predicted effect. np is NumPy. '
 'w contains the full current parameter arrays; all modules may change. f contains ONLY permitted training feedback arrays: '
 'x=noisy x_t, t=timestep, y=target Gaussian epsilon, clean=training clean image, features=current final hidden feature maps for ALL training examples, '
 'prediction=current epsilon prediction, alphabar=32 schedule coefficients. batch.ids/x/t/y identify the paired128-example batch; '
 'layer.<name>.input0/output are full module traces for that batch. generated.samples contains8 fixed-seed DDPM draws for visual diagnosis, not test samples. '
 'Cached features describe the PARENT before the edit; changing earlier layers makes downstream cached features stale. '
 'Statistics, covariance, PCA, linear algebra and closed-form fitting on this training data are allowed. No autograd, backprop, gradient estimation or conventional iterative optimizer training. '
 'NumPy/math and numpy.lib.stride_tricks imports are available. No filesystem/network/private evaluator access. Preserve keys/shapes, finite float32, absolute value<=1000. '
 'Resource limits are4GiB and60CPU seconds per execution; avoid unnecessarily materializing enormous pixel patch matrices. '
 'Return w or {"weights":w,"diagnostics": a small statistics dictionary}. All computations and coefficients are your choice. '
 'Prediction after construction uses ONLY the fixed convolutional forward and saved tensors; the update program never runs during selection or sampling.\n'
 +json.dumps({'feedback_arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in f.items()}}))

class Adapter(common.Adapter):
 def __init__(self,seed,device,folder,compiler,data):
  super().__init__('diffusion',seed,device,folder,compiler,{k:v['x'] for k,v in data.items()},{k:v['y'] for k,v in data.items()});self.data=data
 def evaluate(self,batch,candidate,capture_traces=False):
  split=batch[0]['split'];assert all(b['split']==split for b in batch)
  if capture_traces:assert split=='feedback','Selection traces may never reach reflection'
  ids=np.array([b['index'] for b in batch]);h=self.compiler.key(candidate)
  try:h,state=self.compiler.materialize(candidate);r,traces=forward(state,self.data[split],ids,self.device,capture_traces);error=None
  except Exception as e:error=str(e)[-4000:];r={'mse':float('inf'),'score':-1e30,'scores':np.full(len(ids),-1e30)};traces={}
  common.append(self.folder/'evaluations.jsonl',{'candidate':h,'split':split,'n':len(ids),'capture_traces':capture_traces,'metrics':metrics(r),'error':error,'time':time.time()})
  if split=='feedback':
   for row in self.recent:
    if row.get('child')==h:row.update(measured_feedback_metrics=metrics(r),execution_error=error)
  if capture_traces:self.latest={'h':h,'ids':ids,'traces':traces,'state':None if error else state,'r':r}
  return EvaluationBatch(outputs=[float(x) for x in r['scores']],scores=r['scores'].tolist(),trajectories=[{'split':'feedback','feedback_index':int(i)} for i in ids] if capture_traces else None)
 def make_reflective_dataset(self,candidate,eval_batch,components_to_update):
  assert all(row['split']=='feedback' for row in eval_batch.trajectories)
  return {m:[{'feedback_batch_metrics':metrics(self.latest['r'])}] for m in components_to_update}
 def propose_new_texts(self,candidate,reflective_dataset,components_to_update):
  self.proposals+=1;parent,state=self.compiler.materialize(candidate);assert parent==self.latest['h'];pf=self.folder/'llm'/f'mutation_{self.proposals:03d}'
  try:
   f,packet,manifest=feedback_snapshot(state,self.data['feedback'],self.latest['ids'],self.device,self.seed,self.folder,parent,self.latest['traces']);common.dump(pf/'feedback_manifest.json',manifest)
   source,answer=client.call(instructions(f)+'\n'+json.dumps({'network':spec(state),'feedback':packet,'paired_batch':reflective_dataset[components_to_update[0]],'recent_proposals':self.recent[-4:]}),pf,f'diffusion/{self.seed}/mutation{self.proposals}',system=SYSTEM)
   (pf/'edit.py').write_text(source);updated,report=rich.execute(state,f,source,list(state));common.dump(pf/'program_diagnostics.json',report)
   checkpoint=common.digest({'parent':parent,'source':source,'feedback':manifest['arrays_sha256']});np.savez(self.compiler.folder/f'{checkpoint}.npz',**updated)
   updates={m:json.dumps({'checkpoint':checkpoint}) for m in components_to_update};child=dict(candidate);child.update(updates)
   self.recent.append({'parent':parent,'child':self.compiler.key(child),'hypothesis':answer.split('```')[0][:2000],'program_statistics':report['reported_statistics']});return updates
  except Exception as e:
   common.append(self.folder/'proposal_errors.jsonl',{'proposal':self.proposals,'error':str(e)});self.recent.append({'error':str(e)})
   return {m:json.dumps({'checkpoint':'INVALID_'+str(self.proposals)}) for m in components_to_update}

def run(slug,seed,device,proposals=20,folder_override=None):
 client.configure(slug,compact=True);folder=Path(folder_override) if folder_override is not None else ROOT/'diffusion_runs/models'/slug/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'search_result.json').exists():return
 assert (ROOT/'data/diffusion'/f'seed{seed}/manifest.json').exists(),'Prepare training data/baseline first'
 data=load_data(seed,device);state=core.initial(seed);np.savez(folder/'initial.npz',**state)
 common.dump(folder/'run_config.json',{'model':client.MODELS[slug],'seed':seed,'proposals':proposals,'compact_context':True,'network':spec(state),'source_hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['diffusion_experiment.py','diffusion_core.py','expansion_client.py','rich_worker.py']},'final_test_evaluated':False})
 if (folder/'one_shot.npz').exists():constructed=common.load_npz(folder/'one_shot.npz')
 else:
  ids=np.arange(min(128,len(data['feedback']['x'])));f,packet,manifest=feedback_snapshot(state,data['feedback'],ids,device,seed,folder,'random_initialization');pf=folder/'llm/construction';common.dump(pf/'feedback_manifest.json',manifest);error=None;report={}
  try:
   source,_=client.call(instructions(f)+'\nOne-shot feedback-assisted construction: choose one weight update using this permitted training data.\n'+json.dumps({'network':spec(state),'feedback':packet}),pf,f'diffusion/{seed}/construction',system=SYSTEM)
   (folder/'construction.py').write_text(source);constructed,report=rich.execute(state,f,source,list(state))
  except Exception as e:constructed=state;error=str(e)
  np.savez(folder/'one_shot.npz',**constructed);common.dump(folder/'construction_status.json',{'error':error,'diagnostics':report});del f
 assert not json.loads((ROOT/'runtime/usage.json').read_text())['blocked'],'Global budget stop'
 one,_=forward(constructed,data['selection'],np.arange(len(data['selection']['x'])),device);common.dump(folder/'one_shot_selection.json',metrics(one))
 compiler=rich.Compiler(BACKEND,'diffusion',constructed,folder/'checkpoints');candidate={m:json.dumps({'checkpoint':'base'}) for m in MODULES};adapter=Adapter(seed,device,folder,compiler,data)
 result=gepa.optimize(seed_candidate=candidate,trainset=[{'split':'feedback','index':i} for i in range(len(data['feedback']['x']))],valset=[{'split':'selection','index':i} for i in range(len(data['selection']['x']))],adapter=adapter,candidate_selection_strategy='pareto',module_selector='all',reflection_minibatch_size=128,skip_perfect_score=False,use_merge=False,stop_callbacks=[MaxCandidateProposalsStopper(proposals+20),lambda state:common.mutation_api_calls(folder)>=proposals,lambda state:json.loads((ROOT/'runtime/usage.json').read_text())['blocked']],max_metric_calls=1000+(proposals+20)*1256+100,run_dir=str(folder/'gepa'),seed=seed,track_best_outputs=False,raise_on_exception=True)
 assert not json.loads((ROOT/'runtime/usage.json').read_text())['blocked'],'Global budget stop'
 h,best=compiler.materialize(result.best_candidate);np.savez(folder/'gepa_best.npz',**best);out,_=forward(best,data['selection'],np.arange(len(data['selection']['x'])),device)
 common.dump(folder/'search_result.json',{'seed':seed,'model':slug,'one_shot':metrics(one),'gepa':metrics(out),'best_candidate':h,'proposals':common.mutation_api_calls(folder),'gepa_iterations':adapter.proposals,'metric_calls':result.total_metric_calls,'final_test_evaluated':False});common.dump(folder/'gepa_result.json',result.to_dict())
 from classification_followup import save_prefixes
 save_prefixes(folder,[b for b in [0,5,10,20] if b<=proposals])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=list(client.MODELS),required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--proposals',type=int,default=20);a=p.parse_args();torch.set_num_threads(2);run(a.model,a.seed,a.device,a.proposals)
