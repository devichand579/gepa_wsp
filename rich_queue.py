"""One task queue; seeds run in separate processes with the same frozen protocol."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main(task,device):
 frozen=json.loads((ROOT/'runtime/RICH_FROZEN.json').read_text())
 for name,sha in frozen['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
 for seed in [0,1,2]:
  folder=ROOT/'rich_runs/rich_main'/task/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
  prior=(ROOT/'text_runs/sst2' if task=='sst2' else ROOT/'runs/main'/task)/f'seed{seed}'
  for name in ['adam_best.npz','adam_result.json']:
   if not (folder/name).exists():shutil.copy2(prior/name,folder/name)
  result=folder/'search_result.json'
  if result.exists() and json.loads(result.read_text())['proposals']==30:continue
  if result.exists():result.rename(folder/f'search_result_partial_{int(time.time())}.json')
  stop=folder/'gepa/gepa.stop'
  if stop.exists():stop.unlink()
  with (ROOT/'runtime'/f'rich_{task}_seed{seed}.log').open('a') as log:
   subprocess.run([sys.executable,str(ROOT/'rich_experiment.py'),'--task',task,'--seed',str(seed),'--device',device,'--proposals','30'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  assert json.loads(result.read_text())['proposals']==30
 (ROOT/'runtime'/f'rich_{task}_complete.json').write_text(json.dumps({'task':task,'time':time.time()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--device',required=True);a=p.parse_args();main(a.task,a.device)
