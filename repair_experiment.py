"""Fresh method-corrected conditions; original runs and all spending stay preserved."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import torch
import experiment as common
import rich_experiment as rich
import expansion_client as client
from classification_followup import save_prefixes
ROOT=common.ROOT

def run(job_id,device):
 config=json.loads((ROOT/'runtime/EXPERIMENT_REPAIRS.json').read_text())[job_id]
 folder=ROOT/config['folder'];folder.mkdir(parents=True,exist_ok=True);original=ROOT/config['original']
 common.dump(folder/'repair_provenance.json',{**config,'time':time.time(),'original_result_sha256':hashlib.sha256((original/'search_result.json').read_bytes()).hexdigest(),'basis':'Method compliance only; original results preserved; no final-test results available.'})
 if config['task']=='diffusion':
  import diffusion_experiment as diffusion
  diffusion.run(config['model'],config['seed'],device,config['budget'],folder_override=folder)
 else:
  for name in ['adam_best.npz','adam_result.json']:
   if not (folder/name).exists():shutil.copy2(original/name,folder/name)
  client.configure(config['model'],compact=False)
  rich.run(config['task'],config['seed'],device,folder,config['budget']);save_prefixes(folder,[0,5,10,30,60])
 assert json.loads((folder/'search_result.json').read_text())['proposals']==config['budget']
 common.dump(folder/'FOLLOWUP_COMPLETE.json',{'time':time.time(),'budget':config['budget'],'method_corrected':True})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--model');p.add_argument('--kind');a=p.parse_args();torch.set_num_threads(2);run(a.job,a.device)
