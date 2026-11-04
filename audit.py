"""Audit recorded API messages without printing any secret or dataset content."""
import json,re
from pathlib import Path
from experiment import ROOT,dump

def audit():
    keypath=Path('/tmp/gepa-weight-space-fireworks.key')
    key=keypath.read_text().strip() if keypath.exists() else ''
    checked=0
    for path in (ROOT/'runs').rglob('request.json'):
        raw=path.read_text()
        assert not key or key not in raw, f'Credential found in {path}'
        request=json.loads(raw)
        assert {'model','messages','max_tokens','temperature'}<=set(request)
        assert set(request)<={'model','messages','max_tokens','temperature','reasoning_effort','thinking','response_format','tools','tool_choice'}
        assert not ('thinking' in request and 'reasoning_effort' in request)
        assert [m['role'] for m in request['messages']]==['system','user']
        user=request['messages'][1]['content']
        assert 'selection_index' not in user and 'test_index' not in user
        assert '"split": "selection"' not in user and '"split": "test"' not in user
        ids=[int(x) for x in re.findall(r'"feedback_index"\s*:\s*(\d+)',user)]
        assert all(0<=i<2000 for i in ids)
        checked+=1
    # Scan project artifacts for the credential but only report a boolean/count.
    scanned=0
    for folder in [ROOT/'runs',ROOT/'runtime']:
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.jsonl','.md','.py','.log','.txt']:
                assert not key or key.encode() not in path.read_bytes(),f'Credential in artifact {path}'
                scanned+=1
    result={'requests_checked':checked,'artifacts_secret_scanned':scanned,'passed':True,
            'limitations':'Structural provenance checks and secret scan; content provenance is enforced by the adapter and sandbox, not inferred by searching for every possible encoded test example.'}
    dump(ROOT/'runtime'/'audit.json',result);print(json.dumps(result))

if __name__=='__main__':audit()
