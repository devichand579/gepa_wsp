/* Frozen result exploration only. No requests, trackers, dependencies or model calls. */
"use strict";
(() => {
  const D = window.STUDY;
  if (!D) return;
  const $ = id => document.getElementById(id);
  const names = {mnist:"MNIST",fashion:"Fashion-MNIST",sst2:"SST-2",deepseek:"DeepSeek",minimax:"MiniMax",nemotron:"Nemotron"};
  const mean = xs => xs.reduce((a,b)=>a+b,0)/xs.length;
  const pct = x => (100*x).toFixed(2)+"%";
  const money = x => "$"+x.toFixed(4);
  const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const taskSummary = (group,task,model) => D.classification_summary.filter(r=>r.group===group&&r.task===task&&r.model===model).sort((a,b)=>a.budget-b.budget);
  const adam = task => mean(D.baselines.filter(r=>r.task===task).map(r=>r.accuracy));
  let budgetTask="mnist", modelTask="mnist", diffModel="deepseek";
  let predictionTask="mnist", predictionPage=0;
  let lossTask='mnist';
  const budgets=[0,5,10,30,60];

  function wireButtons(attribute, callback) {
    document.querySelectorAll(`[${attribute}]`).forEach(button=>button.addEventListener("click",()=>{
      document.querySelectorAll(`[${attribute}]`).forEach(b=>b.setAttribute("aria-pressed",String(b===button)));
      callback(button.getAttribute(attribute));
    }));
  }

  function drawBudget() {
    const rows=taskSummary("budget",budgetTask,"deepseek");
    const budget=budgets[Number($("budget-slider").value)], selected=rows.find(r=>r.budget===budget), start=rows[0];
    const raw=D.classification.filter(r=>r.group==="budget"&&r.task===budgetTask), baseline=adam(budgetTask);
    const W=640,H=345,p={l:46,r:20,t:26,b:48};
    const low=Math.max(0,Math.floor((Math.min(...raw.map(r=>r.accuracy*100),...rows.map(r=>(r.accuracy_mean-r.accuracy_sd)*100))-7)/10)*10);
    const high=100;
    const x = b => p.l+(W-p.l-p.r)*b/60;
    const y = v => p.t+(H-p.t-p.b)*(high-v)/(high-low);
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="budget-svg-title budget-svg-desc"><title id="budget-svg-title">${names[budgetTask]} held-out accuracy by mutation budget</title><desc id="budget-svg-desc">Mean over seeds 0, 1, 2, with sample standard deviations. Adam mean ${pct(baseline)}. Selected budget ${budget}, accuracy ${pct(selected.accuracy_mean)}.</desc>`;
    svg+=`<text x="${p.l}" y="12" font-family="system-ui" font-size="10" fill="#667066">Held-out accuracy (%)</text>`;
    for(let v=low;v<=high;v+=10){svg+=`<line x1="${p.l}" x2="${W-p.r}" y1="${y(v)}" y2="${y(v)}" stroke="#d7dacd"/><text x="${p.l-9}" y="${y(v)+4}" text-anchor="end" font-size="10" fill="#667066" font-family="system-ui">${v}</text>`;}
    svg+=`<line x1="${x(budget)}" x2="${x(budget)}" y1="${p.t}" y2="${H-p.b}" stroke="#205c4b" stroke-width="16" opacity=".055"/>`;
    const ya=y(baseline*100);
    svg+=`<line x1="${p.l}" x2="${W-p.r}" y1="${ya}" y2="${ya}" stroke="#b65337" stroke-width="1.6" stroke-dasharray="5 5"/><text x="${W-p.r}" y="${ya-7}" font-size="10" text-anchor="end" fill="#b65337" font-family="system-ui">Adam ${pct(baseline)}</text>`;
    rows.forEach(r=>{const cy=y(r.accuracy_mean*100),cx=x(r.budget),top=y(Math.min(100,100*(r.accuracy_mean+r.accuracy_sd))),bottom=y(Math.max(low,100*(r.accuracy_mean-r.accuracy_sd)));svg+=`<path d="M${cx} ${top}V${bottom}M${cx-4} ${top}H${cx+4}M${cx-4} ${bottom}H${cx+4}" stroke="#205c4b" opacity=".45" fill="none"/>`;});
    if($("show-seeds").checked)raw.forEach(r=>{svg+=`<circle cx="${x(r.budget)+(r.seed-1)*4}" cy="${y(r.accuracy*100)}" r="3" fill="#205c4b" opacity=".25"><title>Seed ${r.seed}, ${r.budget} mutations: ${pct(r.accuracy)}</title></circle>`;});
    svg+=`<path d="${rows.map((r,i)=>`${i?'L':'M'}${x(r.budget)} ${y(r.accuracy_mean*100)}`).join(' ')}" stroke="#205c4b" stroke-width="2.6" fill="none"/>`;
    rows.forEach(r=>{svg+=`<circle cx="${x(r.budget)}" cy="${y(r.accuracy_mean*100)}" r="${r.budget===budget?6:4}" fill="${r.budget===budget?'#205c4b':'#fcfaf4'}" stroke="#205c4b" stroke-width="2"><title>${r.budget} mutations: ${pct(r.accuracy_mean)}, SD ${(100*r.accuracy_sd).toFixed(2)} pp</title></circle><text x="${x(r.budget)}" y="${H-p.b+22}" text-anchor="middle" font-size="10" font-family="system-ui" fill="#667066">${r.budget}</text>`;});
    svg+=`<text x="${(W+p.l-p.r)/2}" y="${H-3}" text-anchor="middle" fill="#667066" font-family="system-ui" font-size="10">Mutation requests after one-shot construction</text></svg>`;
    $("budget-chart").innerHTML=svg;
    $("budget-chart").setAttribute("aria-label",`${names[budgetTask]} budget trajectory. ${budget} mutations: ${pct(selected.accuracy_mean)} mean accuracy; Adam ${pct(baseline)}.`);
    $("budget-label").textContent=budget===0?"ONE-SHOT CONSTRUCTION":`${budget} MUTATIONS`;
    $("budget-accuracy").innerHTML=(100*selected.accuracy_mean).toFixed(2)+"<span>%</span>";
    $("budget-sd").textContent=`±${(selected.accuracy_sd*100).toFixed(2)} percentage points across seeds`;
    const gain=100*(selected.accuracy_mean-start.accuracy_mean);
    $("budget-gain").textContent=(gain>=0?"+":"")+gain.toFixed(2)+" pp";
    $("budget-adam").textContent=pct(baseline);
    $("budget-cost").textContent=money(selected.mean_cumulative_api_cost);
    $("budget-output").textContent=budget+" calls";
    $("budget-slider").setAttribute("aria-valuetext",budget+" mutation requests");
  }

  function drawModels() {
    $("model-cards").innerHTML=["deepseek","minimax","nemotron"].map(model=>{
      const rows=taskSummary("models",modelTask,model),start=rows[0],end=rows.find(r=>r.budget===60);
      const failures=D.reliability.filter(r=>r.group==="models"&&r.task===modelTask&&r.model===model&&!r.construction_succeeded).length;
      return `<article class="model-card"><h3>${names[model]}</h3><span class="fine">60-mutation held-out mean</span><div class="model-value">${pct(end.accuracy_mean)}</div><div class="micro-bar" role="img" aria-label="One shot ${pct(start.accuracy_mean)}; 60 mutations ${pct(end.accuracy_mean)}"><span class="end" style="width:${end.accuracy_mean*100}%"></span><span class="start" style="left:${start.accuracy_mean*100}%"></span></div><div class="micro-labels"><span>0%</span><span>100%</span></div><p>One shot <b>${pct(start.accuracy_mean)}</b> · SD at 60: ${(end.accuracy_sd*100).toFixed(2)} pp</p><p class="cost">${money(end.mean_cumulative_api_cost)} / seed</p><p class="warning">${failures?`${failures}/3 one-shot constructions failed; random weights retained.`:"All three one-shot constructions passed validation."}</p></article>`;
    }).join("");
  }

  function drawDiffusion() {
    const W=750,H=355,p={l:54,r:25,t:43,b:50};
    const colors={deepseek:'#205c4b',minimax:'#b65337',nemotron:'#5e738d'};
    const x=b=>p.l+(W-p.l-p.r)*b/60;
    const y=v=>p.t+(H-p.t-p.b)*(Math.log10(30)-Math.log10(v))/(Math.log10(30)-Math.log10(.05));
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="diff-title diff-desc"><title id="diff-title">Held-out diffusion denoising MSE by mutation budget</title><desc id="diff-desc">Log scale, lower is better. Mean over three seeds, with individual seed dots. At 60 mutations: DeepSeek 0.25518, MiniMax 0.50622, Nemotron 0.53842. Adam mean 0.09161.</desc><text x="${p.l}" y="15" font-family="system-ui" font-size="12" fill="#667066">Noise-prediction MSE · log scale · lower is better</text>`;
    [.1,1,10].forEach(v=>{svg+=`<line x1="${p.l}" x2="${W-p.r}" y1="${y(v)}" y2="${y(v)}" stroke="#d7dacd"/><text x="${p.l-10}" y="${y(v)+4}" text-anchor="end" font-family="system-ui" font-size="12" fill="#667066">${v}</text>`;});
    Object.entries(colors).forEach(([model,color],i)=>{
      const rows=D.diffusion_summary.filter(r=>r.model===model).sort((a,b)=>a.budget-b.budget);
      D.diffusion.filter(r=>r.model===model).forEach(r=>{svg+=`<circle cx="${x(r.budget)+(r.seed-1)*3}" cy="${y(r.mse)}" r="2.7" fill="${color}" opacity=".3"><title>${names[model]}, seed ${r.seed}, ${r.budget} mutations: ${r.mse.toFixed(5)}</title></circle>`;});
      svg+=`<path d="${rows.map((r,i)=>`${i?'L':'M'}${x(r.budget)} ${y(r.mse_mean)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
      rows.forEach(r=>{svg+=`<circle cx="${x(r.budget)}" cy="${y(r.mse_mean)}" r="4" fill="${color}"><title>${names[model]}, ${r.budget} mutations, mean ${r.mse_mean.toFixed(5)}</title></circle>`;});
      const lx=p.l+i*145;
      svg+=`<line x1="${lx}" x2="${lx+17}" y1="32" y2="32" stroke="${color}" stroke-width="2.5"/><text x="${lx+23}" y="36" font-family="system-ui" font-size="11" fill="#667066">${names[model]}</text>`;
    });
    const baseline=mean(D.baselines.filter(r=>r.task==='diffusion').map(r=>r.mse));
    svg+=`<path d="M${p.l} ${y(baseline)}H${W-p.r}" fill="none" stroke="#6b6253" stroke-width="1.6" stroke-dasharray="5 5"/><text x="${W-p.r}" y="${y(baseline)-8}" text-anchor="end" font-size="11" fill="#6b6253" font-family="system-ui">Adam ${baseline.toFixed(5)}</text>`;
    [0,5,10,20,30,60].forEach(b=>{svg+=`<text x="${x(b)}" y="${H-p.b+23}" text-anchor="middle" font-family="system-ui" font-size="12" fill="#667066">${b}</text>`;});
    svg+=`<text x="${W/2}" y="${H-4}" text-anchor="middle" font-size="12" fill="#667066" font-family="system-ui">Mutation requests after one-shot construction</text></svg>`;
    $('diffusion-chart').innerHTML=svg;
  }

  function drawSamples() {
    const seed=Number($("diff-seed").value), rows=D.diffusion.filter(r=>r.model===diffModel&&r.seed===seed);
    const one=rows.find(r=>r.budget===0),end=rows.find(r=>r.budget===60),baseline=D.baselines.find(r=>r.task==="diffusion"&&r.seed===seed);
    $("sample-grids").innerHTML=[[one,"One shot / 0",names[diffModel]],[end,"After 60 mutations",names[diffModel]],[baseline,"Adam baseline","Adam"]].map(([r,title,label])=>`<figure><h3>${title}</h3><img src="assets/${esc(r.generation.grid.split('/').pop())}" width="360" height="360" alt="${label}, seed ${seed}, ${title}: first 64 fixed draws"><div class="sample-mse">Seed ${seed} MSE: ${r.mse.toFixed(5)}</div><figcaption>${label} · fixed first 64 draws</figcaption></figure>`).join("");
    $("sample-download").href="../results/generations/"+end.generation.samples.split("/").pop();
    $("sample-download").setAttribute("download","");
  }

  function drawPredictions() {
    const task=window.PREDICTIONS.tasks[predictionTask];
    const [group,model]=$('prediction-condition').value.split('/');
    const seed=Number($('prediction-seed').value);
    const condition=task.conditions.find(r=>r.group===group&&r.model===model&&r.seed===seed);
    const methods=[['one_shot','One shot'],['gepa',`GEPA · ${condition.budget}`],['adam','Adam']];
    $('prediction-scores').innerHTML=methods.map(([key,title])=>`<div><span>${title}</span><strong>${pct(condition[key].accuracy)}</strong><small>Full held-out split · seed ${seed}</small></div>`).join('');
    $('prediction-context').textContent=predictionTask==='sst2'?'872-example labeled validation split. Original review text is shown; the target receives lowercase token IDs, truncated to 64 tokens, using its training-only vocabulary.':'10,000-example official test split. Images are original 28 × 28 inputs. True labels appear above each method’s answer.';
    $('prediction-construction').hidden=!condition.construction_failed;
    $('prediction-construction').textContent='This one-shot construction failed validation and retained random initialization. GEPA gains include recovery from that failure.';
    const filter=$('prediction-filter').value;
    const selected=task.samples.filter((sample,i)=>{
      const one=condition.one_shot.predictions[i],gepa=condition.gepa.predictions[i],adam=condition.adam.predictions[i];
      if(filter==='disagree')return new Set([one,gepa,adam]).size>1;
      if(filter==='repair')return one!==sample.label&&gepa===sample.label;
      if(filter==='regression')return one===sample.label&&gepa!==sample.label;
      return true;
    });
    const size=predictionTask==='sst2'?4:8;
    predictionPage=Math.min(predictionPage,Math.max(0,Math.ceil(selected.length/size)-1));
    const offset=predictionPage*size;
    $('prediction-cards').classList.toggle('text-predictions',predictionTask==='sst2');
    $('prediction-cards').innerHTML=selected.slice(offset,offset+size).map(sample=>{
      const i=sample.index;
      const input=sample.image?`<img class="prediction-input" src="${sample.image}" alt="${names[predictionTask]} test example ${i}, true class ${esc(task.labels[sample.label])}" width="84" height="84">`:`<blockquote>${esc(sample.text)}</blockquote>`;
      return `<article class="prediction-card" data-index="${i}" data-label="${sample.label}"><div class="prediction-example-id">EXAMPLE ${String(i).padStart(2,'0')}</div><div class="prediction-input-row">${input}<div class="prediction-truth"><span>True label</span><b>${esc(task.labels[sample.label])}</b></div></div><div class="prediction-answers">${methods.map(([key,title])=>{
        const predicted=condition[key].predictions[i],correct=predicted===sample.label;
        return `<div class="prediction-answer ${correct?'correct':'incorrect'}" data-method="${key}" data-prediction="${predicted}"><span>${title}</span><strong>${esc(task.labels[predicted])}</strong><i aria-label="${correct?'Correct':'Incorrect'}">${correct?'✓':'×'}</i></div>`;
      }).join('')}</div></article>`;
    }).join('')||'<p class="prediction-empty">No examples in the fixed 24 match this filter.</p>';
    $('prediction-prev').disabled=predictionPage===0;
    $('prediction-next').disabled=offset+size>=selected.length;
    $('prediction-count').textContent=selected.length?`${offset+1}–${Math.min(offset+size,selected.length)} of ${selected.length}${filter==='all'?' fixed examples':' matching examples (within 24)'}`:'0 matching examples (within 24)';
  }

  function drawLosses() {
    const diffusion=lossTask==='diffusion';
    const original=$('loss-view').querySelector('option[value="budget"]');
    original.disabled=diffusion;
    if(diffusion)$('loss-view').value='matched';
    const view=$('loss-view').value;
    const group=diffusion?'diffusion':view==='matched'?'models':'budget';
    const rows=window.LOSSES.jobs.filter(r=>r.task===lossTask&&r.group===group);
    const budgets=rows[0].points.map(p=>p.budget).sort((a,b)=>a-b);
    const file=`loss_${lossTask}_${view}.png`;
    const asset=window.BLOG_ASSETS?.[file]||'assets/'+file;
    $('loss-image').src=asset;
    const continuation=rows.every(r=>r.observations?.length);
    const observations=rows[0].observations?.length;
    $('loss-image').alt=`${names[lossTask]||'MNIST diffusion'} validation loss: ${continuation?'all three LLMs through 60 mutations, including per-mutation continuation observations':'original DeepSeek frozen prefixes'}, alongside three Adam selected-learning-rate histories; separate native x-axes.`;
    $('loss-download').href=asset;
    $('loss-download').setAttribute('download',file);
    $('loss-context').textContent=(diffusion?'Diffusion, all three LLMs':view==='matched'?'Matched compact-prompt classification, all three LLMs':'Original-display DeepSeek classification')+(continuation?` · ${observations} measured points per LLM and seed, including every mutation ${diffusion?'21':'11'}–60`:` · frozen mutation prefixes ${budgets.join(' / ')}`)+' · seeds 0/1/2 · Adam: three seed curves · validation only.';
    const metric=diffusion?'MSE':'cross-entropy';
    const ends=[...new Set(rows.map(r=>r.model))].map(model=>{
      const values=rows.filter(r=>r.model===model).map(r=>r.points.reduce((a,b)=>a.budget>b.budget?a:b).loss);
      return `${names[model]} ${mean(values).toFixed(4)}`;
    });
    const baselines=[0,1,2].map(seed=>window.LOSSES.baselines[`${lossTask}/${seed}`].selected_loss);
    $('loss-summary').textContent=`Endpoint mean validation ${metric}: ${ends.join(' · ')} · Adam selected checkpoints ${mean(baselines).toFixed(4)}.`;
  }

  function reliability() {
    const groups=[["budget","Original 60-call classification"],["models","Matched 60-call classification"],["diffusion","60-call diffusion"]];
    $("reliability-bars").innerHTML=groups.map(([group,name])=>{
      const rows=D.reliability.filter(r=>r.group===group),n=rows.reduce((s,r)=>s+r.mutation_budget,0),valid=rows.reduce((s,r)=>s+r.valid_mutation_programs,0),fail=rows.filter(r=>!r.construction_succeeded).length;
      return `<div class="reliability-row"><div><span>${name}</span><b>${valid} / ${n}</b></div><div class="track" role="img" aria-label="${valid} of ${n} programs valid"><div class="fill" style="width:${100*valid/n}%"></div></div><small>${fail} failed one-shot construction${fail===1?'':'s'} / ${rows.length}</small></div>`;
    }).join("");
    const totals=new Map();
    D.reliability.forEach(r=>{
      const key=[r.group,r.task,r.model].join('/');
      if(!totals.has(key))totals.set(key,{group:r.group,task:r.task,model:r.model,valid:0,n:0,fail:0,errors:{}});
      const t=totals.get(key);t.valid+=r.valid_mutation_programs;t.n+=r.mutation_budget;t.fail+=!r.construction_succeeded;
      Object.entries(r.failure_counts).forEach(([k,n])=>t.errors[k]=(t.errors[k]||0)+n);
    });
    $("reliability-table").innerHTML=[...totals.values()].sort((a,b)=>[a.group,a.task,a.model].join('/').localeCompare([b.group,b.task,b.model].join('/'))).map(r=>`<tr><td>${r.group} / ${names[r.task]||r.task}</td><td>${names[r.model]}</td><td>${r.valid}/${r.n}</td><td>${r.fail}/3</td><td>${Object.entries(r.errors).map(([k,v])=>esc(k.replaceAll('_',' '))+": "+v).join("; ")}</td></tr>`).join("");
  }

  function filterTable() {
    const group=$("filter-group").value,task=$("filter-task").value,model=$("filter-model").value;
    let n=0;
    document.querySelectorAll("#all-results tr").forEach(r=>{
      r.hidden=![['group',group],['task',task],['model',model]].every(([key,v])=>v==="all"||r.dataset[key]===v);
      if(!r.hidden)n++;
    });
    $("row-count").textContent=n+" row"+(n===1?"":"s");$("empty-results").hidden=n!==0;
  }
  wireButtons("data-budget-task",task=>{budgetTask=task;drawBudget();});
  wireButtons("data-model-task",task=>{modelTask=task;drawModels();});
  wireButtons("data-diff-model",model=>{diffModel=model;drawSamples();});
  wireButtons('data-prediction-task',task=>{predictionTask=task;predictionPage=0;drawPredictions();});
  wireButtons('data-loss-task',task=>{lossTask=task;drawLosses();});
  $('loss-view').addEventListener('change',drawLosses);
  ['prediction-condition','prediction-seed','prediction-filter'].forEach(id=>$(id).addEventListener('change',()=>{predictionPage=0;drawPredictions();}));
  $('prediction-prev').addEventListener('click',()=>{predictionPage--;drawPredictions();});
  $('prediction-next').addEventListener('click',()=>{predictionPage++;drawPredictions();});
  $("budget-slider").addEventListener("input",drawBudget);
  $("show-seeds").addEventListener("change",drawBudget);
  $("diff-seed").addEventListener("change",drawSamples);
  ["filter-group","filter-task","filter-model"].forEach(id=>$(id).addEventListener("change",filterTable));
  document.addEventListener("scroll",()=>{
    const total=document.documentElement.scrollHeight-window.innerHeight;
    $("reading-progress").style.width=(total>0?100*window.scrollY/total:0)+"%";
  },{passive:true});
  drawBudget();drawModels();drawDiffusion();drawSamples();drawPredictions();drawLosses();reliability();filterTable();
})();
