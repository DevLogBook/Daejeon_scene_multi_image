"""Split official UDIS training pairs by connected image content; never use testing."""
import argparse, hashlib, json, random
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--udis-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--val-ratio',type=float,default=.1); p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if not 0<args.val_ratio<1: p.error('val-ratio must be in (0,1)')
    root=args.udis_root.resolve(); training=root/'training'
    if not training.is_dir(): raise FileNotFoundError(f'Expected official training/input1 and input2 under {root}')
    a,b=training/'input1',training/'input2'
    images=lambda folder:{f.stem:f for f in sorted(folder.iterdir()) if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')}
    aa,bb=images(a),images(b)
    if aa.keys()!=bb.keys(): raise ValueError('Unpaired input1/input2 filenames')
    parent={}; records=[]
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    for key in aa:
        pa,pb=aa[key],bb[key]; ha,hb=[hashlib.sha256(x.read_bytes()).hexdigest() for x in (pa,pb)]
        parent[find(ha)]=find(hb); records.append((pa,pb,ha,hb))
    groups={}
    for r in records: groups.setdefault(find(r[2]),[]).append(r)
    keys=list(groups); random.Random(args.seed).shuffle(keys)
    if len(keys)<2: raise ValueError('Need at least two independent image groups')
    selected=set(keys[:max(1,min(len(keys)-1,round(len(keys)*args.val_ratio)))])
    splits={'train':[], 'val':[]}
    for key,rows in groups.items(): splits['val' if key in selected else 'train'].extend(rows)
    args.output.mkdir(parents=True,exist_ok=True)
    for split,rows in splits.items():
        text=''.join(f'{r[0]}\t{r[1]}\n' for r in rows); path=args.output/(split+'_pairs.tsv')
        if path.exists() and path.read_text(encoding='utf-8')!=text: raise FileExistsError(f'Different split exists: {path}')
        path.write_text(text,encoding='utf-8')
    report={'seed':args.seed,'official_subset':'training','split_unit':'connected SHA256 image groups',
            'counts':{s:len(r) for s,r in splits.items()},'root':str(root)}
    (args.output/'splits.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report),flush=True)

if __name__=='__main__': main()
