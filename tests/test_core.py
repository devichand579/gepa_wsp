"""Synthetic checks for the final harness. No datasets or paid API calls."""
from pathlib import Path
import ast
import json
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import core
import classification
import diffusion
import diffusion_model
import llm
import prefixes
import text_model


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_networks_and_forward_traces(self):
        for task, count in [('mnist', 101770), ('fashion', 20490), ('sst2', 83266)]:
            backend = core if task != 'sst2' else types.SimpleNamespace(Network=text_model.Network)
            state = core.initial(task, 0) if task != 'sst2' else text_model.initial(task, 0)
            self.assertEqual(sum(v.size for v in state.values()), count)
            x = np.zeros((4, 1, 28, 28), np.float32) if task != 'sst2' else np.ones((4, 64), np.int64)
            y = np.zeros(4, np.int64)
            result, traces = classification.forward(backend, task, state, x, y, 'cpu', True)
            expected = (core if task != 'sst2' else text_model).evaluate_state(task, state, x, y, 'cpu', True)
            np.testing.assert_allclose(result['probs'], expected['probs'], atol=1e-6)
            self.assertTrue(traces)
        state = diffusion_model.initial(0)
        self.assertEqual(sum(v.size for v in state.values()), 56609)
        clean = np.zeros((4, 1, 16, 16), np.float32)
        x, t, y = diffusion_model.fixed_batch(clean, 81000, 'cpu')
        data = dict(clean=clean, x=x.numpy(), t=t.numpy(), y=y.numpy())
        measured, traces = diffusion.forward(state, data, np.arange(4), 'cpu', True)
        expected = diffusion_model.evaluate(state, clean, 81000, 'cpu')
        self.assertAlmostEqual(measured['mse'], expected['mse'], places=6)
        self.assertTrue(traces)

    def test_sandbox_and_gradient_restriction(self):
        source = (Path(__file__).resolve().parents[1]/'src/sandbox.py').read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'parameter_gradient_reasons')
        ns = {'ast': ast}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<screen>', 'exec'), ns)
        screen = ns['parameter_gradient_reasons']
        self.assertTrue(screen(ast.parse("def edit(w,f):\n X=f['features']; e=f['prediction']-f['y']; w['weight']-=.01*(X.T@e); return w")))
        self.assertFalse(screen(ast.parse("def edit(w,f):\n X=f['features']; w['weight']=np.linalg.solve(X.T@X+np.eye(X.shape[1]),X.T@f['y']); return w")))
        state = core.initial('mnist', 0)
        features = np.zeros((4, 128), np.float32)
        features[:2, 0] = 1
        features[2:, 1] = 1
        feedback = dict(features=features, y=np.array([0, 0, 1, 1]))
        program = "def edit(w,f):\n X=f['features']; Y=np.eye(10)[f['y']]; w['fc2.weight'][:]=np.linalg.solve(X.T@X+np.eye(128),X.T@Y).T; return w\n"
        updated, report = classification.execute(state, feedback, program, ['fc2.weight', 'fc2.bias'])
        self.assertTrue(np.all(updated['fc2.weight'][:2, :2].diagonal() > 0))
        self.assertEqual(report['program_executions'], 2)
        np.testing.assert_array_equal(updated['fc1.weight'], state['fc1.weight'])
        with self.assertRaises(ValueError):
            classification.execute(state, feedback, 'import os\ndef edit(w,f):\n return w', list(state))

    def test_selection_isolation_and_gepa(self):
        import gepa
        from gepa.utils.stop_condition import MaxCandidateProposalsStopper
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            state = {k:np.zeros_like(v) for k, v in core.initial('mnist', 0).items()}
            state['fc2.bias'][1] = 4
            compiler = classification.Compiler(core, 'mnist', state, folder/'checkpoints')
            candidate = {m:json.dumps({'checkpoint':'base'}) for m in core.modules('mnist')}
            class Adapter(classification.Adapter):
                def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
                    self.proposals += 1
                    assert all(r['split'] == 'feedback' for r in self.latest_batch)
                    updated = {k:v.copy() for k, v in self.latest['state'].items()}
                    updated['fc2.bias'][:] = 0
                    updated['fc2.bias'][0] = 4
                    child = 'a'*64
                    np.savez(self.compiler.folder/f'{child}.npz', **updated)
                    return {m:json.dumps({'checkpoint':child}) for m in components_to_update}
                def make_reflective_dataset(self, candidate, batch, components):
                    self.latest_batch = batch.trajectories
                    return super().make_reflective_dataset(candidate, batch, components)
            x = np.zeros((8, 1, 28, 28), np.float32)
            y = np.zeros(8, np.int64)
            adapter = Adapter(core, 'mnist', 0, 'cpu', folder, compiler, dict(feedback=x, selection=x), dict(feedback=y, selection=y))
            with self.assertRaises(AssertionError):
                adapter.evaluate([dict(split='selection', index=0)], candidate, True)
            result = gepa.optimize(seed_candidate=candidate, trainset=[dict(split='feedback', index=i) for i in range(8)], valset=[dict(split='selection', index=i) for i in range(8)], adapter=adapter, module_selector='all', reflection_minibatch_size=8, skip_perfect_score=False, stop_callbacks=[MaxCandidateProposalsStopper(1)], max_metric_calls=100, run_dir=str(folder/'gepa'))
            self.assertEqual(result.num_candidates, 2)
            self.assertGreater(max(result.val_aggregate_scores), .9)
            records = [json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines()]
            self.assertTrue(all(not r['capture_traces'] for r in records if r['split'] == 'selection'))

    def test_prefix_and_billing_boundaries(self):
        with tempfile.TemporaryDirectory() as td, patch.object(core, 'ROOT', Path(td)):
            root = Path(td)
            core.dump(root/'runtime/usage.json', dict(calls=[], spent=0., reserved=0., cap=1., max_calls=3, blocked=False))
            budget = core.Budget()
            index = budget.reserve('synthetic', 10, 10)
            llm.settle_failed_dispatch(index, 'synthetic provider failure')
            ledger = json.loads(budget.path.read_text())
            self.assertEqual(ledger['calls'][0]['status'], 'conservative_settlement')
            self.assertGreater(ledger['spent'], 0)
            self.assertAlmostEqual(ledger['reserved'], 0)
            folder = root/'run'
            core.dump(folder/'llm/mutation_001/dispatch.json', dict(ledger_call=0, time=20))
            core.dump(folder/'llm/mutation_002/dispatch.json', dict(ledger_call=1, time=40))
            core.append(folder/'evaluations.jsonl', dict(split='selection', time=10, candidate='initial', metrics=dict(score=.2)))
            core.append(folder/'evaluations.jsonl', dict(split='selection', time=30, candidate='first', metrics=dict(score=.4)))
            core.append(folder/'evaluations.jsonl', dict(split='selection', time=50, candidate='future', metrics=dict(score=.9)))
            self.assertEqual(prefixes.select_prefix(folder, 0)['candidate'], 'initial')
            self.assertEqual(prefixes.select_prefix(folder, 1)['candidate'], 'first')

    def test_fresh_searches_without_provider_calls(self):
        """Run both real search engines with local deterministic proposal fixtures."""
        for task in ['mnist', 'diffusion']:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                core.dump(root/'runtime/usage.json', dict(calls=[], spent=0., reserved=0., cap=1., max_calls=10, blocked=False))
                data = root/'data'/task/'seed0'
                data.mkdir(parents=True)
                for split in ['feedback', 'selection']:
                    if task == 'mnist':
                        np.savez(data/(split+'.npz'), x=np.zeros((8, 1, 28, 28), np.float32), y=np.zeros(8, np.int64))
                    else:
                        np.savez(data/(split+'.npz'), clean=np.zeros((8, 1, 16, 16), np.float32))
                core.dump(data/'manifest.json', dict(seed=0))
                folder = root/'run'
                count = [0]
                def propose(prompt, destination, label, **kwargs):
                    count[0] += 1
                    core.dump(destination/'dispatch.json', dict(ledger_call=count[0], time=time.time()))
                    constructing = destination.name == 'construction'
                    if task == 'mnist':
                        index = 1 if constructing else 0
                        source = f"def edit(w,f):\n w['fc2.weight'][:]=0; w['fc2.bias'][:]=0; w['fc2.bias'][{index}]=4; return w\n"
                    else:
                        value = 2 if constructing else 0
                        source = f"def edit(w,f):\n w['out.weight'][:]=0; w['out.bias'][:]={value}; return w\n"
                    return source, 'Local test fixture'
                with patch.object(core, 'ROOT', root), patch.object(classification, 'ROOT', root), patch.object(diffusion, 'ROOT', root), patch.object(classification, 'call', propose), patch.object(llm, 'call', propose):
                    if task == 'mnist':
                        classification.run(task, 0, 'cpu', folder, 1)
                    else:
                        diffusion.run('deepseek', 0, 'cpu', 1, folder_override=folder)
                    prefixes.save_prefixes(folder, [0, 1])
                result = json.loads((folder/'search_result.json').read_text())
                self.assertEqual(result['proposals'], 1)
                self.assertGreater(result['gepa']['score'], result['one_shot']['score'])
                self.assertTrue((folder/'prefixes/budget001.npz').exists())


if __name__ == '__main__':
    unittest.main()
