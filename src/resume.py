"""Continue preserved GEPA searches with a shared cap and observational loss logs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

SOURCES = ['experiment.py', 'rich_experiment.py', 'rich_worker.py', 'text_experiment.py',
           'strict_experiment.py', 'tensor_dsl.py', 'tensor_dsl_worker.py', 'expansion_client.py',
           'diffusion_core.py', 'diffusion_experiment.py', 'classification_followup.py']
PREFIXES = [0, 5, 10, 20, 30, 60]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(archive, workspace, cap):
    archive, workspace = Path(archive).resolve(), Path(workspace).resolve()
    plan_path = workspace/'runtime/PLAN.json'
    if plan_path.exists():
        plan = read(plan_path)
        assert plan['additional_cap'] == cap and plan['archive'] == str(archive)
        return plan
    assert archive != workspace and cap > 0
    workspace.mkdir(parents=True, exist_ok=True)
    ledger = read(archive/'runtime/usage.json')
    assert abs(ledger['reserved']) < 1e-9 and not ledger['blocked']
    assert not any(c['status'] in ['pending', 'unknown_billing'] for c in ledger['calls'])
    baseline_spending, baseline_calls = ledger['spent'], len(ledger['calls'])
    ledger.update(cap=baseline_spending+cap, max_calls=10000)
    write(workspace/'runtime/usage.json', ledger)
    for name in SOURCES:
        shutil.copy2(archive/name, workspace/name)
    for task in ['mnist', 'fashion', 'sst2', 'diffusion']:
        for seed in [0, 1, 2]:
            source = archive/'data'/task/f'seed{seed}'
            destination = workspace/'data'/task/f'seed{seed}'
            destination.mkdir(parents=True, exist_ok=True)
            for name in ['feedback.npz', 'selection.npz', 'manifest.json', 'vocab.json']:
                if (source/name).exists():
                    shutil.copy2(source/name, destination/name)
            baseline = archive/f'diffusion_runs/baselines/seed{seed}' if task == 'diffusion' else archive/f'rich_runs/rich_main/{task}/seed{seed}'
            target = workspace/baseline.relative_to(archive)
            target.mkdir(parents=True, exist_ok=True)
            for name in ['adam_best.npz', 'adam_result.json']:
                shutil.copy2(baseline/name, target/name)
    results = read(archive/'expanded_results/results.json')
    jobs = []
    for row in results['classification']+results['diffusion']:
        if row['group'] not in ['models', 'diffusion']:
            continue
        prior = 20 if row['group'] == 'diffusion' else 10
        if row['budget'] != prior:
            continue
        source = (archive/row['weights']).parent.parent
        folder = workspace/source.relative_to(archive)
        assert (source/'gepa/gepa_state.bin').is_file()
        if folder.exists():
            assert not (folder/'CONTINUATION.json').exists(), 'Unexpected partial preparation'
            shutil.rmtree(folder)
        shutil.copytree(source, folder)
        original_prefixes = {str(n):read(folder/'prefixes'/f'budget{n:03d}.json') for n in PREFIXES if n <= prior}
        source_requests = sorted((source/'llm').glob('mutation_*/request.json'))
        request = read(source_requests[-1])
        config = dict(group=row['group'], model=row['model'], task=row['task'], seed=row['seed'],
                      prior=prior, budget=60, folder=str(folder.relative_to(workspace)),
                      source=str(source), initial_state_sha256=sha(source/'gepa/gepa_state.bin'),
                      source_result_sha256=sha(source/'search_result.json'),
                      original_prefixes=original_prefixes, system=request['messages'][0]['content'])
        config['id'] = f"{row['group']}_{row['model']}_{row['task']}_{row['seed']}"
        for name in ['search_result.json', 'run_config.json', 'gepa_result.json']:
            if (folder/name).exists():
                (folder/name).rename(folder/(Path(name).stem+'_before_extension.json'))
        (folder/'gepa/gepa.stop').unlink(missing_ok=True)
        write(folder/'CONTINUATION.json', config)
        jobs.append(config)
    assert len(jobs) == 36
    first = ['models_deepseek_mnist_0', 'models_minimax_mnist_0', 'models_nemotron_mnist_0', 'diffusion_deepseek_diffusion_0']
    jobs.sort(key=lambda j:(first.index(j['id']) if j['id'] in first else 4,
                            {'deepseek':0, 'minimax':1, 'nemotron':2}[j['model']], j['task'], j['seed']))
    plan = dict(time=time.time(), archive=str(archive), workspace=str(workspace), additional_cap=cap,
                historical_spending=baseline_spending, historical_calls=baseline_calls,
                total_cap=baseline_spending+cap, jobs=jobs,
                prices={'deepseek':[.30, 1.20], 'minimax':[.15, .60], 'nemotron':[.05, .20]},
                prefixes=PREFIXES, source_hashes={name:sha(workspace/name) for name in SOURCES},
                runner_sha256=sha(__file__), followup='Existing final test results are already known. This is a declared follow-up, not a fresh confirmatory held-out study.',
                resume='Native GEPA0.1.4 restores candidate/evaluation/adapter state; library sampler RNG restarts from the experiment seed, so this is not a bit-identical uninterrupted trajectory.',
                information_boundary='Only feedback and selection files are copied. No final-test files, predictions, or prior reports enter reflection. Loss logging is observational; final evaluation requires selection freeze and source review.')
    write(plan_path, plan)
    return plan


def complete(workspace, job):
    folder = workspace/job['folder']
    result = folder/'search_result.json'
    if not result.exists() or read(result)['proposals'] != 60:
        return False
    for n in PREFIXES:
        path = folder/'prefixes'/f'budget{n:03d}.json'
        if not path.exists() or not path.with_suffix('.npz').exists():
            return False
        assert sha(path.with_suffix('.npz')) == read(path)['weights_sha256']
        if n <= job['prior']:
            assert read(path) == job['original_prefixes'][str(n)]
    return True


def run_job(workspace, job_id, device):
    workspace = Path(workspace).resolve()
    plan = read(workspace/'runtime/PLAN.json')
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    for name, digest in plan['source_hashes'].items():
        assert sha(workspace/name) == digest
    assert sha(__file__) == plan['runner_sha256']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(workspace))
    import numpy as np
    import torch
    import gepa
    from gepa.core.state import GEPAState
    from gepa.strategies.eval_policy import FullEvaluationPolicy
    import experiment as common
    import rich_experiment as rich
    import expansion_client as client
    import classification_followup as prefixes
    import diffusion_experiment as diffusion
    torch.set_num_threads(2)
    folder = workspace/job['folder']
    if complete(workspace, job):
        return
    partial = folder/'search_result.json'
    if partial.exists():
        partial.rename(folder/f'search_result_partial_{time.time_ns()}.json')
    (folder/'gepa/gepa.stop').unlink(missing_ok=True)
    saved = GEPAState.load(str(folder/'gepa'))
    prior_calls = common.mutation_api_calls(folder)
    assert job['prior'] <= prior_calls <= 60
    assert saved.adapter_state['proposals'] >= job['prior']
    seed_hash = rich.arrays_digest(common.load_npz(folder/'one_shot.npz'))
    parents = {common.digest(dict(seed_weights=seed_hash, numeric_components=c)) for c in saved.program_candidates}
    write(folder/f'resume_state_{prior_calls:03d}.json', dict(time=time.time(), mutations=prior_calls,
          adapter_proposals=saved.adapter_state['proposals'], iteration=saved.i,
          candidates=len(saved.program_candidates), state_sha256=sha(folder/'gepa/gepa_state.bin')))
    for slug, prices in plan['prices'].items():
        client.MODELS[slug]['prices'] = prices
    rich.SYSTEM = job['system']
    diffusion.SYSTEM = job['system']
    original_call = client.call

    def guarded_call(prompt, destination, label, **kwargs):
        destination = Path(destination)
        assert not (workspace/'runtime/GLOBAL_ATTENTION.json').exists(), 'Continuation requires attention'
        assert destination.name.startswith('mutation_'), 'Construction must be reused'
        if not (folder/'FIRST_RESUME_PROPOSAL.json').exists():
            assert int(destination.name.split('_')[1]) > job['prior']
            parent = read(destination/'feedback_manifest.json')['parent_checkpoint']
            assert parent in parents, 'First resumed proposal did not use the saved pool'
            write(folder/'FIRST_RESUME_PROPOSAL.json', dict(time=time.time(), proposal=destination.name,
                  parent=parent, loaded_candidates=len(parents), old_budget=job['prior'], model=job['model']))
        return original_call(prompt, destination, label, **kwargs)

    client.call = guarded_call
    client.configure(job['model'], compact=True)
    old_prefixes = prefixes.save_prefixes

    def preserve_prefixes(destination, budgets):
        old_prefixes(destination, [n for n in budgets if n > job['prior']])
    prefixes.save_prefixes = preserve_prefixes
    original_optimize = gepa.optimize

    def measured_optimize(*args, **kwargs):
        adapter = kwargs['adapter']
        last = {}
        class LossLog:
            def record(self, state, accepted=None):
                index = FullEvaluationPolicy().get_best_program(state)
                checkpoint, weights = adapter.compiler.materialize(state.program_candidates[index])
                cached = checkpoint == last.get('checkpoint')
                if not cached:
                    if job['task'] == 'diffusion':
                        metrics = {split:diffusion.metrics(diffusion.forward(weights, adapter.data[split], np.arange(len(adapter.data[split]['x'])), device)[0]) for split in ['feedback', 'selection']}
                    else:
                        metrics = {split:common.summary(rich.forward(adapter.b, job['task'], weights, adapter.x[split], adapter.y[split], device)[0]) for split in ['feedback', 'selection']}
                    last.update(checkpoint=checkpoint, metrics=metrics)
                common.append(folder/'loss_curve.jsonl', dict(time=time.time(), mutation_calls=common.mutation_api_calls(folder),
                              iteration=state.i+1, candidate=checkpoint, paired_proposal_accepted=accepted,
                              cached_metrics=cached, **last['metrics']))
            def on_iteration_start(self, event):
                if not last:
                    self.record(event['state'])
            def on_iteration_end(self, event):
                self.record(event['state'], event['proposal_accepted'])
        kwargs['callbacks'] = [*(kwargs.get('callbacks') or []), LossLog()]
        kwargs['stop_callbacks'] = [*(kwargs.get('stop_callbacks') or []), lambda state:(workspace/'runtime/GLOBAL_ATTENTION.json').exists()]
        return original_optimize(*args, **kwargs)
    gepa.optimize = measured_optimize
    if job['task'] == 'diffusion':
        diffusion.run(job['model'], job['seed'], device, 60, folder_override=folder)
    else:
        rich.run(job['task'], job['seed'], device, folder, 60)
    preserve_prefixes(folder, PREFIXES)
    assert complete(workspace, job), 'Search stopped before required budget'
    write(folder/'COMPLETE.json', dict(time=time.time(), budget=60, final_test_evaluated=False))


def freeze(workspace, plan):
    path = workspace/'runtime/SELECTIONS_FROZEN.json'
    if path.exists():
        return
    write(path, dict(time=time.time(), selections={j['id']:{str(n):read(workspace/j['folder']/'prefixes'/f'budget{n:03d}.json') for n in PREFIXES} for j in plan['jobs']}))
    write(workspace/'runtime/SOURCE_REVIEW_REQUIRED.json', dict(time=time.time(), reason='All36 continuation searches finished. Inspect all selected program lineages before final evaluation.', final_test_access=False))


def worker(workspace, device):
    workspace = Path(workspace).resolve()
    plan = read(workspace/'runtime/PLAN.json')
    for job in plan['jobs']:
        if (workspace/'runtime/GLOBAL_ATTENTION.json').exists():
            return
        locks = workspace/'runtime/locks'
        locks.mkdir(exist_ok=True)
        with (locks/(job['id']+'.lock')).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if complete(workspace, job):
                continue
            folder = workspace/job['folder']
            for attempt in range(4):
                if read(workspace/'runtime/usage.json')['blocked']:
                    return
                if (folder/'llm/provider_nonretryable.json').exists():
                    write(workspace/'runtime/GLOBAL_ATTENTION.json', dict(job=job['id'], reason='Provider rejected configuration; further jobs paused'))
                    return
                marker = folder/'llm/provider_circuit.json'
                if marker.exists():
                    while time.time() < read(marker)['retry_after']:
                        time.sleep(min(30, max(0, read(marker)['retry_after']-time.time())))
                status = workspace/'runtime'/('job_'+job['id']+'.json')
                write(status, dict(time=time.time(), job=job['id'], phase='running', device=device, attempt=attempt, worker_pid=os.getpid()))
                with (workspace/'runtime'/(job['id']+'.log')).open('a') as log:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'job', '--workspace', str(workspace), '--job', job['id'], '--device', device], cwd=workspace, stdout=log, stderr=subprocess.STDOUT)
                if complete(workspace, job):
                    write(status, dict(time=time.time(), job=job['id'], phase='complete', device=device))
                    break
                if result.returncode and not marker.exists():
                    write(status, dict(time=time.time(), job=job['id'], phase='attention', device=device, returncode=result.returncode))
                    if (folder/'llm/provider_nonretryable.json').exists():
                        write(workspace/'runtime/GLOBAL_ATTENTION.json', dict(job=job['id'], reason='Provider configuration rejected; further dispatch paused'))
                    return
            else:
                write(workspace/'runtime/GLOBAL_ATTENTION.json', dict(job=job['id'], reason='Bounded recovery attempts exhausted'))
                return
        if (workspace/'runtime/GLOBAL_ATTENTION.json').exists():
            return
    if all(complete(workspace, j) for j in plan['jobs']):
        with (workspace/'runtime/completion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            freeze(workspace, plan)


def status(workspace):
    workspace = Path(workspace).resolve()
    plan = read(workspace/'runtime/PLAN.json')
    ledger = read(workspace/'runtime/usage.json')
    rows = []
    for j in plan['jobs']:
        folder = workspace/j['folder']
        calls = {read(p)['ledger_call'] for p in (folder/'llm').glob('mutation_*/dispatch.json')}
        phase_path = workspace/'runtime'/('job_'+j['id']+'.json')
        rows.append(dict(job=j['id'], mutation_calls=len(calls), target=60,
                         phase=read(phase_path)['phase'] if phase_path.exists() else 'queued',
                         resume_verified=(folder/'FIRST_RESUME_PROPOSAL.json').exists()))
    return dict(jobs=rows, completed=sum(r['phase']=='complete' for r in rows),
                additional_requests=len(ledger['calls'])-plan['historical_calls'],
                additional_spent=ledger['spent']-plan['historical_spending'], reserved=ledger['reserved'],
                additional_cap=plan['additional_cap'], cumulative_spent=ledger['spent'], blocked=ledger['blocked'])


def start(archive, workspace, cap, workers=4):
    plan = prepare(archive, workspace, cap)
    for i in range(workers):
        name = f'gepa_sixty_{i}'
        if subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode == 0:
            continue
        command = shlex.join([sys.executable, str(Path(__file__).resolve()), 'worker', '--workspace', plan['workspace'], '--device', f'cuda:{i%2}'])
        subprocess.run(['tmux', 'new-session', '-d', '-s', name, command], check=True)
    return status(workspace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'worker', 'job', 'status'])
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--cap', type=float, default=40)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--job')
    args = parser.parse_args()
    if args.command == 'start':
        print(json.dumps(start(args.archive, args.workspace, args.cap, args.workers), indent=2))
    elif args.command == 'worker':
        worker(args.workspace, args.device)
    elif args.command == 'job':
        run_job(args.workspace, args.job, args.device)
    else:
        print(json.dumps(status(args.workspace), indent=2))
