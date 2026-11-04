"""Generate figures and a factual blog draft after final evaluation is complete."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT,load_npz

def report():
    final=json.loads((ROOT/'runs'/'main'/'FINAL_RESULTS.json').read_text())
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    labels=['One-shot LLM','LLM + GEPA','Adam'];methods=['one_shot','gepa','adam']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,task in zip(axes,['mnist','fashion']):
        rows=[r for r in final['runs'] if r['task']==task]
        values=np.array([[r['accuracy'][m]*100 for m in methods] for r in rows])
        ax.bar(range(3),values.mean(0),color=['#8299ac','#247da8','#e7a447'],alpha=.8)
        for i,row in enumerate(values):ax.scatter(np.arange(3)+(i-1)*.06,row,color='black',s=18,zorder=3)
        ax.set_xticks(range(3),labels,rotation=15);ax.set_title(task.upper());ax.set_ylim(0,100)
    axes[0].set_ylabel('Final test accuracy (%)');fig.suptitle('Three seeds; dots show individual runs')
    fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,['mnist','fashion']):
        for seed in [0,1,2]:
            folder=ROOT/'runs'/'main'/task/f'seed{seed}'
            rows=[json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]
            rows=[r for r in rows if r['split']=='selection' and not r['error']]
            best=-float('inf');selected=[]
            for r in rows:
                if r['metrics']['score']>best:
                    best=r['metrics']['score'];acc=r['metrics']['accuracy']
                selected.append(acc*100)
            ax.plot(range(len(selected)),selected,marker='.',label=f'Seed {seed}')
        ax.set_title(task.upper());ax.set_xlabel('Full selection-set evaluations');ax.legend()
    axes[0].set_ylabel('Selected candidate accuracy (%)');fig.tight_layout();fig.savefig(out/'selection_curves.png',dpi=180);plt.close(fig)
    base=ROOT/'runs'/'main'/'fashion'/'seed0'
    a=load_npz(base/'one_shot.npz')['conv1.weight'];b=load_npz(base/'gepa_best.npz')['conv1.weight']
    fig,axes=plt.subplots(2,16,figsize=(14,2.5))
    scale=max(float(np.abs(a).max()),float(np.abs(b).max()),1e-6)
    for row,weights in enumerate([a,b]):
        for j in range(16):axes[row,j].imshow(weights[j,0],cmap='coolwarm',vmin=-scale,vmax=scale);axes[row,j].axis('off')
    fig.suptitle('CNN first-layer filters: one-shot (top), GEPA (bottom), seed 0; all filters shown')
    fig.tight_layout();fig.savefig(out/'cnn_filters.png',dpi=180);plt.close(fig)
    table=['| Task | Seed | Construction valid | One-shot | GEPA | Adam | GEPA gain |','|---|---:|---|---:|---:|---:|---:|']
    reliability=[]
    for r in final['runs']:
        folder=ROOT/'runs'/'main'/r['task']/f'seed{r["seed"]}'
        status=json.loads((folder/'construction_status.json').read_text());valid=status['error'] is None
        validity=('yes' if status.get('fully_valid',valid) else 'partial' if valid else 'no; random fallback')
        a=r['accuracy'];table.append(f'| {r["task"]} | {r["seed"]} | {validity} | {a["one_shot"]:.1%} | {a["gepa"]:.1%} | {a["adam"]:.1%} | {r["paired_gain"]*100:+.2f} pp |')
        evals=[json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]
        invalid={e['candidate'] for e in evals if e.get('error')}
        errors=folder/'proposal_errors.jsonl'
        generation_errors=len(errors.read_text().splitlines()) if errors.exists() else 0
        reliability.append({'task':r['task'],'seed':r['seed'],'construction_valid':valid,'construction_fully_valid':status.get('fully_valid',valid),'construction_operations':status.get('operations'),'construction_error':status['error'],
                            'invalid_evaluated_candidates':len(invalid),'generation_errors':generation_errors})
    ledger=json.loads((ROOT/'runtime'/'usage.json').read_text())
    frozen=json.loads((ROOT/'runtime'/'PROTOCOL_FROZEN.json').read_text())
    text='''# Can an LLM learn neural weights through reflective evolutionary search?

We tested direct neural-weight construction and refinement on two image
classification tasks. The optimizing LLM received a fixed tensor-edit interface;
GEPA selected and evaluated its proposed parameter edits. The LLM submitted bounded declarative JSON operations; no generated Python executed. No gradients or
conventional numerical fitting routines were used in the LLM conditions. Adam
was trained independently as a conventional baseline.

This is an exploratory study with three seeds, not a demonstration that direct
weight programming replaces gradient training.

## Models and protocol

- MNIST: a 784–128–10 ReLU MLP with 101,770 parameters.
- Fashion-MNIST: a two-convolution CNN with 20,490 parameters.
- Each seed: 2,000 balanced feedback/training examples, 1,000 disjoint selection
  examples, and the untouched official 10,000-example test set.
- One-shot construction saw ten illustrative labeled images. GEPA started from
  that exact checkpoint and received up to 40 additional mutation proposals.
  This additional feedback is part of the treatment, not an equal-compute comparison.
- Adam used the same 2,000-example pool, three learning rates, up to 50 epochs per
  rate and validation-based checkpoint selection. All tuning is counted.

The selected checkpoint maximizes mean `0.9 * correctness + 0.1 * p(correct class)`.
The official GEPA engine provides instance-wise Pareto selection; our adapter
evolves module tensor-edit programs, cycling through a joint all-layer edit and each individual layer. Crossover is disabled. The network forward
function stays fixed; edit programs are never executed during prediction.

## Results

'''+ '\n'.join(table)+'''

![Final accuracy](figures/accuracy.png)

![Selection curves](figures/selection_curves.png)

![CNN filters](figures/cnn_filters.png)

## Interpretation and limits

Read the individual-seed results before drawing a conclusion. Changes in a few
seeds do not establish robust superiority. Adam and reflective search have
different data exposure, priors and computation. A greedy-reflection control is
needed to attribute any gain specifically to GEPA's Pareto selection.

Per-run paired test-example intervals and exact McNemar values are in
`runs/main/FINAL_RESULTS.json`. These intervals are conditional on a selected
model and do not quantify all search/training randomness. Reusing the same test
set across seeds does not create independent test datasets.

## Information boundary and failures

Only feedback examples and feedback-derived diagnostics entered the LLM context.
Selection examples and labels stayed inside the evaluator, and all main model
selections were frozen before test inference. The edit namespace contained only
Python, NumPy, a fixed worker and tensor inputs; it had no dataset, credentials,
network or access to the project. Structural audits and artifact secret scans
are recorded in `runtime/audit.json`.

Earlier Python-program experiments allowed internal feature computation and analytical fitting despite the intended restrictions. They are preserved separately under `runs/exploratory_python/` and are excluded from these strict results. The replacement interpreter cannot express network execution, matrix products or analytical fitting.

Invalid operations are rejected individually while valid operations still execute. Partially valid construction submissions are marked explicitly; no generated submission was manually repaired.

All pilot failures are retained. Initial MiniMax calls exhausted their output
budget on reasoning, and another completed response described a construction
without submitting executable edits. Prompt/interface revisions were made before the main
protocol freeze; they are not counted as successful learning results. Invalid
main submissions consume their proposal budget and are reported in run artifacts.

## Reproduction and provenance

See `STRICT_PROTOCOL.md`, `README.md`, the historical plan, and the frozen protocol.
Run transcripts, dependency versions, model checkpoints, lineage, and usage are
saved locally. Credentials are excluded. Transformer and diffusion experiments
are separate follow-up phases and are not implied by these two results.

The motivation comes from [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
and [GEPA](https://arxiv.org/abs/2507.19457); this experiment tests their combination
on different tasks and models rather than reproducing their headline claims.
'''
    text+='\nA failed construction leaves the random initialization in place; the table explicitly marks those cases. GEPA then searches from that same fallback. Gains from these runs include recovery from a failed submission and must not be described as refining a successfully constructed classifier.\n'
    text+='\nGenerated-program reliability:\n\n```json\n'+json.dumps(reliability,indent=2)+'\n```\n'
    text+='\nThese are familiar public datasets. We controlled deliberate feedback and test access; we cannot rule out familiarity or contamination in the reflection LLM’s pretraining.\n'
    text+=f'\nFrozen reflection configuration: `{frozen.get("reflection_model")}`, effort `{frozen.get("reasoning_effort")}`.\n'
    text+=f'\nAPI ledger: {len(ledger["calls"])} recorded requests, ${ledger["spent"]:.4f} conservatively estimated from uncached token rates, including pilots.\n'
    (ROOT/'BLOG.md').write_text(text)
    print('Generated BLOG.md and figures; interpretation still requires review.')

if __name__=='__main__':report()
