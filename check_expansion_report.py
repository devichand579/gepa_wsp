"""Render the complete report schema with synthetic numbers; no model/test access."""
import csv,tempfile,re
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import expansion_finalize as report

classification=[];diffusion=[];baselines=[]
for task in ['mnist','fashion','sst2']:
 for seed in [0,1,2]:
  baselines.append({'task':task,'seed':seed,'accuracy':.5})
  for budget in [0,5,10,30,60]:classification.append({'group':'budget','task':task,'model':'deepseek','seed':seed,'budget':budget,'accuracy':.5,'cumulative_api_cost':0})
  for model in ['deepseek','minimax','nemotron']:
   for budget in [0,5,10]:classification.append({'group':'models','task':task,'model':model,'seed':seed,'budget':budget,'accuracy':.5,'cumulative_api_cost':0})
for seed in [0,1,2]:
 baselines.append({'task':'diffusion','seed':seed,'mse':.1,'generation':{'grid':'synthetic.png'}})
 for model in ['deepseek','minimax','nemotron']:
  for budget in [0,5,10,20]:diffusion.append({'group':'diffusion','task':'diffusion','model':model,'seed':seed,'budget':budget,'mse':.2,'cumulative_api_cost':0,'generation':{'grid':'synthetic.png'}})
result={'classification_summary':report.aggregate(classification,'accuracy'),'diffusion_summary':report.aggregate(diffusion,'mse'),'diffusion':diffusion,'baselines':baselines,'accounting':{'whole_project_cost':0,'whole_project_requests':0},'audit_review_flags':0}
result['reliability']=[{'group':'diffusion','task':'diffusion','model':'nemotron','seed':seed,'construction_succeeded':False,'mutation_budget':20,'valid_mutation_programs':0,'failure_counts':{'method_rejection':5,'response_truncation':5,'code_or_array_error':10}} for seed in [0,1,2]]
with tempfile.TemporaryDirectory() as td:
 root=Path(td);out=root/'expanded_results';out.mkdir();Image.fromarray(np.zeros((16,16),dtype=np.uint8)).save(root/'synthetic.png')
 with patch.object(report,'ROOT',root),patch.object(report,'OUT',out):report.make_report(result)
 text=(root/'BLOG_EXPANDED.md').read_text()
 assert '| diffusion | diffusion | nemotron | 0/60 | 3 |' in text
 assert 'method_rejection: 15' in text and 'response_truncation: 15' in text,'Full trajectories must be counted once, not once per prefix'
 for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):assert (root/target).exists(),target
 for name in ['budget','models','diffusion']:
  with Image.open(root/'figures/expanded'/f'{name}.png') as img:img.verify()
 with (out/'summary.csv').open() as f:assert len(list(csv.DictReader(f)))==54
 assert all(row['n_seeds']==3 for row in result['classification_summary']+result['diffusion_summary'])
print('PASS: complete162-condition/3-seed report schema,54 summary rows, all plot PNGs and image links; synthetic values only')
