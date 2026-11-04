import tempfile,json,hashlib,time,fcntl
from pathlib import Path
from unittest.mock import patch
import expansion_queue as queue
import expansion_client as client
assert len(list(queue.jobs('classification')))==36 and len(list(queue.jobs('diffusion')))==9
with tempfile.TemporaryDirectory() as td:
 root=Path(td);job={'folder':root,'budget':20,'prefixes':[0,5,10,20]}
 (root/'search_result.json').write_text(json.dumps({'proposals':20}));(root/'prefixes').mkdir()
 assert not queue.verify_job(job),'Missing budget checkpoints accepted'
 for n in job['prefixes']:
  p=root/'prefixes'/f'budget{n:03d}.npz';p.write_bytes(b'numeric-fixture')
  p.with_suffix('.json').write_text(json.dumps({'weights_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
 assert queue.verify_job(job)
 (root/'prefixes/budget005.npz').write_bytes(b'changed')
 try:queue.verify_job(job)
 except AssertionError:pass
 else:raise AssertionError('Corrupted checkpoint accepted')
with tempfile.TemporaryDirectory() as td:
 root=Path(td);llm=root/'llm'
 for i in [1,2,3]:
  p=llm/f'mutation_{i:03d}';p.mkdir(parents=True);(p/'dispatch.json').write_text(json.dumps({'ledger_call':i}));(p/'service_failure.json').write_text('{}')
 target=llm/'mutation_004';target.mkdir()
 try:client.check_provider_circuit(target)
 except RuntimeError:pass
 else:raise AssertionError('Provider circuit failed to stop dispatch')
 assert (root/'gepa/gepa.stop').exists() and (llm/'provider_circuit.json').exists()
 info=json.loads((llm/'provider_circuit.json').read_text());info['retry_after']=time.time()-1;(llm/'provider_circuit.json').write_text(json.dumps(info))
 client.check_provider_circuit(target);assert not (llm/'provider_circuit.json').exists()
with tempfile.TemporaryDirectory() as td,patch.object(queue,'ROOT',Path(td)):
 root=Path(td);(root/'runtime').mkdir();job={'id':'same-condition'}
 with queue.job_lock_path(job).open('a') as owner,patch.object(queue,'_run_job') as launch,patch.object(queue,'verify_job',return_value=True) as verify:
  fcntl.flock(owner,fcntl.LOCK_EX)
  assert queue.run_job(job) is False and queue.job_finished(job) is False
  launch.assert_not_called();verify.assert_not_called()
 with patch.object(queue,'_run_job') as launch,patch.object(queue,'verify_job',return_value=True):
  assert queue.run_job(job) is True and queue.job_finished(job) is True
  launch.assert_called_once_with(job)
print('PASS:45 jobs, prefix/hash checks, circuit cooldown and exclusive job claims; busy prefixes are not read and duplicate workers do not dispatch; no API calls')
