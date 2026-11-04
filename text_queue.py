"""Independent text queue; shared source hashes remain frozen."""
import concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(seed):
 frozen=json.loads((ROOT/'runtime/TEXT_FROZEN.json').read_text())
 for name,expected in frozen['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
 folder=ROOT/'text_runs/sst2'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
 for method,name in [('adam','adam_result.json'),('search','search_result.json')]:
  if (folder/name).exists():continue
  with (ROOT/'runtime'/f'text_{seed}_{method}.log').open('a') as log:
   subprocess.run([sys.executable,str(ROOT/'text_experiment.py'),'--seed',str(seed),'--device','cuda:0','--method',method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('STRICT_CONCURRENCY','3'))) as pool:
  for future in concurrent.futures.as_completed([pool.submit(run,s) for s in [0,1,2]]):future.result()
 subprocess.run([sys.executable,str(ROOT/'text_finalize.py')],cwd=ROOT,check=True)
 (ROOT/'runtime/TEXT_RESULTS_READY.json').write_text(json.dumps({'time':time.time(),'results':'text_runs/FINAL_RESULTS.json'}))
