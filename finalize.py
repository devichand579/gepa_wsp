"""Run only after all main selections are frozen. No API calls in this module."""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from experiment import ROOT, dump, evaluate_state, load_npz

def finalize(device='cuda:0'):
    roots=[ROOT/'runs'/'main'/task/f'seed{seed}' for task in ['mnist','fashion'] for seed in [0,1,2]]
    for root in roots:
        for name in ['search_result.json','adam_result.json','one_shot.npz','gepa_best.npz','adam_best.npz']:
            assert (root/name).exists(),f'Final test locked: missing {root/name}'
    frozen={str(root.relative_to(ROOT)):json.loads((root/'search_result.json').read_text())['best_candidate'] for root in roots}
    dump(ROOT/'runs'/'main'/'SELECTION_FROZEN.json',frozen)
    reports=[]
    for root in roots:
        task=root.parent.name;seed=int(root.name.removeprefix('seed'))
        test=load_npz(ROOT/'data'/'final_test'/f'{task}.npz')
        result={m:evaluate_state(task,load_npz(root/file),test['x'],test['y'],device)
                for m,file in [('one_shot','one_shot.npz'),('gepa','gepa_best.npz'),('adam','adam_best.npz')]}
        a=result['one_shot']['pred']==test['y'];b=result['gepa']['pred']==test['y']
        gained=int((~a & b).sum());lost=int((a & ~b).sum());d=b.astype(float)-a
        se=float(d.std(ddof=1)/math.sqrt(len(d)))
        report={'task':task,'seed':seed,'n_test':len(d),'accuracy':{m:r['accuracy'] for m,r in result.items()},
                'cross_entropy':{m:r['cross_entropy'] for m,r in result.items()},
                'paired_gain':float(d.mean()),'gain_normal_95ci':[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],
                'wrong_to_right':gained,'right_to_wrong':lost,
                'mcnemar_exact_p':float(binomtest(gained,gained+lost,p=.5).pvalue) if gained+lost else 1.}
        np.savez(root/'final_predictions.npz',y=test['y'],**{m:r['pred'] for m,r in result.items()})
        dump(root/'final_test.json',report);reports.append(report)
    aggregate={}
    for task in ['mnist','fashion']:
        rows=[r for r in reports if r['task']==task]
        aggregate[task]={m:{'mean':float(np.mean([r['accuracy'][m] for r in rows])),
                            'sd':float(np.std([r['accuracy'][m] for r in rows],ddof=1))}
                         for m in ['one_shot','gepa','adam']}
        aggregate[task]['gains']=[r['paired_gain'] for r in rows]
    dump(ROOT/'runs'/'main'/'FINAL_RESULTS.json',{'runs':reports,'aggregate':aggregate,
        'uncertainty_note':'Per-run paired intervals concern test examples conditional on that trained model. Across-seed means/SD are exploratory with n=3; no independence claim across repeated test sets.'})
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(2);finalize(a.device)
