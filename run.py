"""One entry point for data preparation, Adam, GEPA search and frozen evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

TASKS = ['mnist', 'fashion', 'sst2', 'diffusion']
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT/'work')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--task', choices=TASKS+['all'], default='all')
    resume = commands.add_parser('resume', help='Continue the original preserved searches')
    resume.add_argument('--archive', type=Path, required=True)
    resume.add_argument('--cap', type=float, required=True, help='Additional API dollar cap above preserved historical spending')
    resume.add_argument('--workers', type=int, default=4)
    commands.add_parser('status', help='Show continuation progress and inclusive accounting')
    for name in ['adam', 'search', 'evaluate']:
        command = commands.add_parser(name)
        command.add_argument('--task', choices=TASKS, required=True)
        command.add_argument('--seed', type=int, choices=[0, 1, 2], default=0)
        command.add_argument('--device', default='cuda:0')
        if name == 'search':
            command.add_argument('--model', choices=['deepseek', 'minimax', 'nemotron'], default='deepseek')
            command.add_argument('--mutations', type=int, default=10)
            command.add_argument('--cap', type=float, required=True, help='Inclusive API dollar cap for this workspace')
            command.add_argument('--full-context', action='store_true', help='Original budget-study display instead of compact display')
        if name == 'evaluate':
            command.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    if args.command in ['resume', 'status']:
        sys.path.insert(0, str(ROOT/'src'))
        from resume import start, status
        result = start(args.archive, args.workspace, args.cap, args.workers) if args.command == 'resume' else status(args.workspace)
        print(json.dumps(result, indent=2))
        return
    os.environ['GEPA_WSP_WORKSPACE'] = str(args.workspace.resolve())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(ROOT/'src'))
    import torch
    import core
    torch.set_num_threads(2)

    if args.command == 'prepare':
        from prepare import prepare_images, prepare_text
        tasks = TASKS if args.task == 'all' else [args.task]
        if 'mnist' in tasks or 'diffusion' in tasks:
            prepare_images('mnist')
        if 'fashion' in tasks:
            prepare_images('fashion')
        if 'sst2' in tasks:
            prepare_text()
        if 'diffusion' in tasks:
            from diffusion_model import prepare
            for seed in [0, 1, 2]:
                prepare(seed)
        return

    folder = core.ROOT/'baselines'/args.task/f'seed{args.seed}'
    if args.command == 'adam':
        folder.mkdir(parents=True, exist_ok=True)
        if args.task == 'diffusion':
            from diffusion_model import adam
            adam(args.seed, args.device)
        elif args.task == 'sst2':
            from text_model import adam
            adam(args.seed, args.device, folder)
        else:
            core.run_adam(args.task, args.seed, args.device, folder)
        return

    if args.command == 'evaluate':
        state = core.load_npz(args.weights)
        if args.task == 'diffusion':
            from torch.nn.functional import interpolate
            from diffusion_model import evaluate
            x = core.load_npz(core.ROOT/'data/final_test/mnist.npz')['x']
            clean = interpolate(torch.from_numpy(x), size=(16, 16), mode='bilinear', align_corners=False).numpy()*2-1
            result = evaluate(state, clean, 83000+args.seed, args.device)
            print(json.dumps({k:result[k] for k in ['mse', 'score']}))
        else:
            from classification import backend
            path = core.ROOT/'data/final_test'/(f'sst2/seed{args.seed}.npz' if args.task == 'sst2' else args.task+'.npz')
            data = core.load_npz(path)
            result = backend(args.task, args.seed).evaluate_state(args.task, state, data['x'], data['y'], args.device)
            print(json.dumps(core.summary(result)))
        return

    if args.mutations < 1 or not 0 < args.cap < float('inf'):
        parser.error('Search requires positive --mutations and a finite positive --cap')
    ledger = core.ROOT/'runtime/usage.json'
    if ledger.exists():
        if json.loads(ledger.read_text())['cap'] != args.cap:
            parser.error('This workspace already has a different cap; its ledger was preserved')
    else:
        core.dump(ledger, dict(calls=[], spent=0., reserved=0., cap=args.cap, max_calls=10000, blocked=False))
    from llm import configure
    from prefixes import save_prefixes
    folder = core.ROOT/'runs'/args.task/args.model/f'seed{args.seed}_budget{args.mutations}'
    configure(args.model, compact=not args.full_context)
    if args.task == 'diffusion':
        if args.full_context:
            parser.error('The diffusion study uses compact context')
        from diffusion import run
        run(args.model, args.seed, args.device, args.mutations, folder_override=folder)
    else:
        from classification import run
        run(args.task, args.seed, args.device, folder, args.mutations)
    save_prefixes(folder, sorted({0, args.mutations, *[n for n in [5, 10, 20, 30, 60] if n <= args.mutations]}))


if __name__ == '__main__':
    main()
