"""Prepare SST-2 without pretrained tokenizers; validation labels stay evaluator-only."""
import collections,hashlib,json,re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from experiment import ROOT,dump

TOKEN=re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]")
def tokenize(text):return TOKEN.findall(text.lower())
def normalized(text):return ' '.join(tokenize(text))
def encode(text,vocab):
    ids=[vocab.get(w,1) for w in tokenize(text)[:64]] or [1]
    return ids+[0]*(64-len(ids))
def prepare():
    raw={}
    for split in ['train','validation']:
        p=hf_hub_download('stanfordnlp/sst2',f'data/{split}-00000-of-00001.parquet',repo_type='dataset')
        raw[split]=pq.read_table(p).to_pydict()
    train=raw['train'];test=raw['validation'];test_keys={normalized(x) for x in test['sentence']}
    seen=set();eligible=[]
    for i,text in enumerate(train['sentence']):
        key=normalized(text)
        if key and key not in test_keys and key not in seen:eligible.append(i);seen.add(key)
    ids=np.array(eligible);labels=np.array(train['label']);audit=[]
    for seed in [0,1,2]:
        rng=np.random.default_rng(41000+seed);feedback=[];selection=[]
        for label in [0,1]:
            pool=ids[labels[ids]==label].copy();rng.shuffle(pool)
            feedback.extend(pool[:2000]);selection.extend(pool[2000:2500])
        rng.shuffle(feedback);rng.shuffle(selection)
        counts=collections.Counter(w for i in feedback for w in tokenize(train['sentence'][i]))
        words=['<pad>','<unk>']+[w for w,n in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0]))[:1998]]
        vocab={w:i for i,w in enumerate(words)};folder=ROOT/'data/sst2'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
        # Keep distinct raw texts that collapse under the small vocabulary out of selection.
        # This is a hash-only input audit; validation predictions are never computed.
        blocked={tuple(encode(train['sentence'][i],vocab)) for i in feedback}
        blocked.update(tuple(encode(text,vocab)) for text in test['sentence'])
        selection=[];feedback_set=set(feedback)
        for label in [0,1]:
            pool=ids[labels[ids]==label].copy();rng.shuffle(pool);count=0
            for i in pool:
                row=tuple(encode(train['sentence'][i],vocab))
                if int(i) in feedback_set or row in blocked:continue
                selection.append(int(i));blocked.add(row);count+=1
                if count==500:break
            assert count==500
        rng.shuffle(selection)
        dump(folder/'vocab.json',{'words':words,'counts':[0,0]+[counts[w] for w in words[2:]],'fit_split':'feedback only'})
        arrays={}
        for split,chosen in [('feedback',feedback),('selection',selection)]:
            arrays[split]=np.array([encode(train['sentence'][i],vocab) for i in chosen],dtype=np.int64)
            np.savez(folder/f'{split}.npz',x=arrays[split],y=labels[chosen],ids=np.array(chosen))
        tx=np.array([encode(text,vocab) for text in test['sentence']],dtype=np.int64)
        final=ROOT/'data/final_test/sst2';final.mkdir(parents=True,exist_ok=True)
        np.savez(final/f'seed{seed}.npz',x=tx,y=np.array(test['label']),ids=np.array(test['idx']))
        def hashes(a):return {hashlib.sha256(row.tobytes()).hexdigest() for row in a}
        sets={k:hashes(a) for k,a in dict(arrays,test=tx).items()}
        row={'seed':seed,'normalized_text_cross_split_duplicates':0,'encoded_feedback_selection_duplicates':len(sets['feedback']&sets['selection']),'encoded_feedback_test_duplicates':len(sets['feedback']&sets['test']),'encoded_selection_test_duplicates':len(sets['selection']&sets['test'])};audit.append(row)
        dump(folder/'manifest.json',{'dataset':'stanfordnlp/sst2','seed':seed,'feedback_ids':feedback,'selection_ids':selection,'final_split':'official validation, 872 examples','vocabulary_size':len(words),'max_length':64,'tokenizer':TOKEN.pattern,'input_duplicate_audit':row,'normalized_train_rows_excluded_or_deduplicated':len(labels)-len(ids)})
    dump(ROOT/'data/sst2/duplicate_audit.json',audit);print(json.dumps(audit))
if __name__=='__main__':prepare()
