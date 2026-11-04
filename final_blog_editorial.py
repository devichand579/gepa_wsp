"""Presentation of frozen results only: no API calls, fitting, or model selection."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
result=json.loads((ROOT/'expanded_results/results.json').read_text())
original=ROOT/'expanded_results/BLOG_GENERATED.md'
if not original.exists():shutil.copy2(ROOT/'BLOG_EXPANDED.md',original)
text=original.read_text()

def mean_baseline(task,metric):
 return float(np.mean([r[metric] for r in result['baselines'] if r['task']==task]))

def summary(group,task,model,budget):
 return next(r for r in result['classification_summary'] if (r['group'],r['task'],r['model'],r['budget'])==(group,task,model,budget))

cost=result['accounting']['whole_project_cost']
intro=f'''We tested whether an LLM can improve fixed neural-network weights through
GEPA's reflection-and-selection loop, without parameter-gradient updates in the
selected weight programs. The complete study cost **${cost:.2f}**, including pilots,
discarded method-violating trajectories and replacements. It covers **45 runs,
three LLMs, three seeds, and 162 nested budget checkpoints**, plus Adam baselines.

The loop improved one-shot constructions, but it did not generally replace Adam.
The following are final held-out means over all three seeds in the DeepSeek
60-proposal study; a proposal budget excludes the initial one-shot request.

| Task | One-shot | GEPA, 30 proposals | GEPA, 60 proposals | Adam |
|---|---:|---:|---:|---:|
'''
for task,label in [('mnist','MNIST MLP'),('fashion','Fashion-MNIST CNN'),('sst2','SST-2 transformer container')]:
 vals=[100*summary('budget',task,'deepseek',b)['accuracy_mean'] for b in [0,30,60]]
 intro+=f'| {label} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | {100*mean_baseline(task,"accuracy"):.2f}% |\n'
intro+='''
The SST-2 result needs a qualification: many successful programs disabled attention
and encoded a bag-of-words classifier in the embeddings and head. This demonstrates
statistical construction inside a transformer-shaped network, not learned attention.
More reflection was not uniformly better: SST-2 fell from 73.51% at 30 proposals to
72.52% at 60, even though selection followed the predeclared development objective.

In the separate matched 10-proposal comparison, MiniMax averaged 84.77% on MNIST,
82.36% on Fashion-MNIST and 72.90% on SST-2; DeepSeek averaged 83.27%, 80.92% and
73.32%. MiniMax cost less per run in this comparison. These are exploratory means,
not evidence of a reliable model ranking. All nine MiniMax one-shot constructions
failed validation, so its large gains also measure recovery from broken submissions.
Nemotron was cheaper, but its SST-2 mean stayed near chance at 53.13%.

Diffusion was the clearest limitation. At 20 proposals, final denoising MSE was
0.27113 for DeepSeek, 0.84117 for MiniMax and 1.00299 for Nemotron, versus 0.09161
for Adam (lower is better). The fixed seed-0 grids below show noisy LLM-produced
samples and more digit-like Adam samples; this is a qualitative illustration,
not a perceptual-quality benchmark.

The practical finding is that reflection can repair or improve statistical weight
programs on small tasks. It does not establish general gradient-free network training,
an advantage from Pareto selection specifically, or a compute-matched win over Adam.

'''
title,body=text.split('\n',1)
text=title+'\n\n'+intro+body.lstrip()

# Keep the original linear/SD plot as an audit artifact. This additional view
# displays every positive observed MSE without negative symmetric error bars.
fig,ax=plt.subplots(figsize=(8,4.8))
for model,label in [('deepseek','DeepSeek'),('minimax','MiniMax'),('nemotron','Nemotron')]:
 rows=sorted([r for r in result['diffusion_summary'] if r['model']==model],key=lambda r:r['budget'])
 line,=ax.plot([r['budget'] for r in rows],[r['mse_mean'] for r in rows],marker='o',label=label)
 points=[r for r in result['diffusion'] if r['model']==model]
 assert all(np.isfinite(r['mse']) and r['mse']>0 for r in points)
 ax.scatter([r['budget'] for r in points],[r['mse'] for r in points],s=18,alpha=.4,color=line.get_color())
ax.axhline(mean_baseline('diffusion','mse'),color='black',linestyle='--',label='Adam mean')
ax.set_yscale('log');ax.set_xticks([0,5,10,20]);ax.set_xlabel('Mutation API calls')
ax.set_ylabel('Held-out denoising MSE (log scale; lower is better)')
ax.set_title('Diffusion: lines show means, faint dots show individual seeds')
ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
fig.savefig(ROOT/'figures/expanded/diffusion_log.png',dpi=180);plt.close(fig)
text=text.replace('![Diffusion denoising](figures/expanded/diffusion.png)',
 '![Diffusion denoising: all seeds on a log scale](figures/expanded/diffusion_log.png)\n\nThe diffusion chart uses a log scale and shows individual seed values; the original linear plot is retained in `figures/expanded/diffusion.png`.')
for before,after in {'uses60':'uses 60','prefixes0,5,10,30,60':'prefixes 0, 5, 10, 30, 60','from30 calls':'from 30 calls','at0,5,10 calls':'at 0, 5, 10 calls','seeds0,1,2':'seeds 0, 1, 2','a101770-parameter':'a 101,770-parameter','a20490-parameter':'a 20,490-parameter','83266-parameter':'83,266-parameter','a56609-parameter':'a 56,609-parameter','fixed32-step':'fixed 32-step','MNIST16x16':'MNIST 16x16','at0,5,10,20 calls':'at 0, 5, 10, 20 calls','fresh60-call':'fresh 60-call','a60-CPU-second':'a 60-CPU-second','by0.9*correct+':'by 0.9*correct +','have10000 examples':'have 10,000 examples','its872-example':'its 872-example'}.items():text=text.replace(before,after)
(ROOT/'BLOG_EXPANDED.md').write_text(text)
notes={'input':'expanded_results/results.json','original_generated_blog':str(original.relative_to(ROOT)),
 'changes':'Editorial findings and clearer positive-MSE diffusion plot only; no changes to checkpoints, selections, inference, metrics, experimental source hashes or cost ledger.'}
(ROOT/'runtime/BLOG_PRESENTATION_NOTES.json').write_text(json.dumps(notes,indent=2))
print('Final blog introduction and diffusion log plot written from frozen results.')
