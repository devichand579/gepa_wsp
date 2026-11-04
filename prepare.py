import argparse, hashlib, json
from pathlib import Path
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path(__file__).resolve().parent

def prepare(task):
    cls = MNIST if task == 'mnist' else FashionMNIST
    raw = ROOT/'data'/'raw'
    try:
        train = cls(raw, train=True, download=True)
        test = cls(raw, train=False, download=True)
    except RuntimeError:
        if task != 'fashion': raise
        cls.mirrors = ['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/']
        train = cls(raw, train=True, download=True)
        test = cls(raw, train=False, download=True)
    x = train.data.numpy().astype('float32')[:,None]/255.
    y = np.asarray(train.targets, dtype='int64')
    for seed in [0,1,2]:
        rng = np.random.default_rng(10000+seed)
        feedback, selection = [], []
        for label in range(10):
            ids = rng.permutation(np.flatnonzero(y == label))
            feedback.extend(ids[:200]); selection.extend(ids[200:300])
        feedback = rng.permutation(feedback); selection = rng.permutation(selection)
        assert not set(feedback)&set(selection)
        dest = ROOT/'data'/task/f'seed{seed}'; dest.mkdir(parents=True,exist_ok=True)
        np.savez(dest/'feedback.npz',x=x[feedback],y=y[feedback],ids=feedback)
        np.savez(dest/'selection.npz',x=x[selection],y=y[selection],ids=selection)
        manifest = {'task':task,'seed':seed,'feedback_ids':feedback.tolist(),'selection_ids':selection.tolist()}
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
    testdir = ROOT/'data'/'final_test'; testdir.mkdir(parents=True,exist_ok=True)
    np.savez(testdir/f'{task}.npz',x=test.data.numpy().astype('float32')[:,None]/255.,y=np.asarray(test.targets,dtype='int64'))
    print(json.dumps({'task':task,'feedback':2000,'selection':1000,'test':len(test),'status':'prepared'}),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['mnist','fashion']);prepare(p.parse_args().task)
