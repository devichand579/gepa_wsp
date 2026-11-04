"""Check test gating, paired statistics, and selected-program provenance without test reads."""
import hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import expansion_finalize as report
import experiment as common

with tempfile.TemporaryDirectory() as td,patch.object(report,'ROOT',Path(td)):
 root=Path(td);(root/'runtime').mkdir()
 with patch.object(common,'load_npz',side_effect=AssertionError('Private test arrays must not be read')):
  try:report.validate_gate()
  except AssertionError as e:assert 'locked' in str(e)
  else:raise AssertionError('Missing selection freeze accepted')
 y=np.array([0,0,1,1]);base=np.array([1,0,1,0]);current=np.array([0,0,1,0]);stats=report.paired_classification(base,current,y)
 assert stats['paired_gain']==.25 and stats['wrong_to_right']==1 and stats['right_to_wrong']==0
 checked={'programs':[{'in_selected_lineage':True,'review_flags':['manual optimizer?'],'source_sha256':'s','source':'toy.py'}]}
 try:report.require_mechanism_review(checked)
 except RuntimeError:pass
 else:raise AssertionError('Flagged selected source bypassed review')
 checked['programs'][0]['review_flags']=[]
 try:report.require_mechanism_review(checked)
 except RuntimeError:pass
 else:raise AssertionError('Unflagged selected source bypassed mandatory source inspection')
 (root/'runtime/EXPANSION_MECHANISM_REVIEW.json').write_text(json.dumps({'s':{'verdict':'allowed_statistics','reason':'Synthetic test decision'}}));report.require_mechanism_review(checked)
 # A known prohibited source must remain blocked even if a naming heuristic misses it.
 checked['programs'][0]['review_flags']=[]
 (root/'runtime/EXPANSION_MECHANISM_REVIEW.json').write_text(json.dumps({'s':{'verdict':'prohibited_optimizer','reason':'Synthetic known optimizer'}}))
 try:report.require_mechanism_review(checked)
 except RuntimeError:pass
 else:raise AssertionError('Known prohibited selected source bypassed review without heuristic flags')
with tempfile.TemporaryDirectory() as td:
 folder=Path(td);(folder/'checkpoints').mkdir();(folder/'prefixes').mkdir();(folder/'llm/mutation_001').mkdir(parents=True)
 (folder/'construction_status.json').write_text('{"error":null}');(folder/'construction.py').write_text('def edit(w,f): return w\n')
 base=common.digest({'base':'toy'});(folder/'checkpoints'/f'{base}.json').write_text(json.dumps({'out':json.dumps({'checkpoint':'base'})}))
 code='def edit(w,f):\n    w["out.bias"][:]=0\n    return w\n';m={'parent_checkpoint':base,'arrays_sha256':'feedback-only'};d=folder/'llm/mutation_001';(d/'edit.py').write_text(code);(d/'feedback_manifest.json').write_text(json.dumps(m));(d/'program_diagnostics.json').write_text('{}')
 child=common.digest({'parent':base,'source':code,'feedback':'feedback-only'});candidate=common.digest({'child':child});(folder/'checkpoints'/f'{candidate}.json').write_text(json.dumps({'out':json.dumps({'checkpoint':child})}));(folder/'prefixes/budget001.json').write_text(json.dumps({'candidate':candidate}))
 hashes=report.selected_programs({'folder':folder,'prefixes':[1]});assert hashes=={hashlib.sha256(code.encode()).hexdigest(),hashlib.sha256((folder/'construction.py').read_bytes()).hexdigest()}
with tempfile.TemporaryDirectory() as td:
 root=Path(td);(root/'llm/mutation_001').mkdir(parents=True)
 (root/'construction_status.json').write_text(json.dumps({'error':'Wrong shape: out.bias'}));(root/'llm/mutation_001/program_diagnostics.json').write_text('{}')
 errors=[{'proposal':2,'error':'METHOD_VIOLATION: unnamed gradient'},{'proposal':3,'error':'Incomplete LLM response: length'},{'proposal':4,'error':'ValueError: incompatible dimensions'},{'proposal':5,'error':'Provider failure'}]
 (root/'proposal_errors.jsonl').write_text('\n'.join(json.dumps(r) for r in errors))
 counts=report.proposal_reliability(root,5)
 assert not counts['construction_succeeded'] and counts['valid_mutation_programs']==1 and counts['failed_mutation_proposals']==4 and counts['unclassified_slots']==0
 assert counts['failure_counts']=={'method_rejection':1,'response_truncation':1,'code_or_array_error':1,'provider_failure':1}
print('PASS: test gate, paired counts, all-selected-source review, numeric lineage and distinct invalid/truncated/provider proposal outcomes; no test data or API calls')
