"""Read-only scene/split audit for a GRM geometry cache; no Torch dependency.

Metadata audit does not prove pixel-level dataset independence or GT accuracy.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_id(value):
    value=str(value)
    return value.zfill(4) if value.isdigit() else value


def audit(preparation,cache=None):
    meta=json.loads(preparation.read_text(encoding='utf-8-sig'))
    inventory=meta.get('inventory',[]); split=meta.get('group_split',{})
    errors=[]; warnings=[]
    if meta.get('recipe')!='v3_depth_cache_v1': errors.append('Unsupported or missing preparation recipe')
    if meta.get('size')!=512: errors.append('This plan requires the existing 512 cache')
    names=[(r['domain'],r['scene']) for r in inventory]
    inventory_groups={(r['domain'],r['scene']):r['group'] for r in inventory}
    if len(set(names))!=len(names): errors.append('Duplicate inventory scene identities')
    groups={r['group'] for r in inventory}
    if groups!=set(split): errors.append('Inventory/group_split keys differ')
    if any(s not in ('train','val') for s in split.values()): errors.append('Unexpected split name')
    forbidden=sorted({scene_id(r['scene']) for r in inventory if r['domain']=='megadepth'
                      and scene_id(r['scene']) in {'0015','0022'}})
    if forbidden: errors.append('MegaDepth1500 scenes included in cache inventory: '+','.join(forbidden))
    counts={}
    for domain in ('megadepth','uav'):
        counts[domain]={}
        for subset in ('train','val'):
            records=[r for r in inventory if r['domain']==domain and split.get(r['group'])==subset]
            counts[domain][subset]=dict(scenes=len(records),frames=sum(r.get('frames',0) for r in records),
                groups=sorted({r['group'] for r in records}),scene_ids=sorted(r['scene'] for r in records))
        if not counts[domain]['train']['scenes'] or not counts[domain]['val']['scenes']:
            errors.append(f'{domain} needs both train and val scenes')
    if len(counts['uav']['train']['groups'])==1:
        warnings.append('UAV training has one field group; more pairs/dates do not create independent fields')
    manifests={}; records={}; manifest_checked=False
    if cache is not None:
        manifest_checked=True
        for subset in ('train','val'):
            file=cache/subset/'manifest.jsonl'
            if not file.is_file():
                errors.append(f'Missing manifest: {file}');records[subset]=[];continue
            data=[json.loads(s) for s in file.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
            records[subset]=data
            ids=[r['pair_id'] for r in data]
            if len(set(ids))!=len(ids): errors.append(f'Duplicate pair IDs in {subset}')
            support=Counter(); domain_counts=Counter(); scene_counts=Counter()
            for r in data:
                domain_counts[r['domain']]+=1;scene_counts[r['domain']+'/'+r['scene']]+=1
                if inventory_groups.get((r['domain'],r['scene']))!=r['group']:
                    errors.append(f'Manifest scene/group absent from inventory: {r["pair_id"]}')
                if r['split']!=subset or split.get(r['group'])!=subset:
                    errors.append(f'Split disagreement: {r["pair_id"]}')
                if r['domain']=='megadepth' and scene_id(r['scene']) in {'0015','0022'}:
                    errors.append(f'MegaDepth test scene in manifest: {r["pair_id"]}')
                if min(r.get('trusted_ab',0),r.get('trusted_ba',0))<meta.get('min_points',32):
                    errors.append(f'Insufficient trusted points: {r["pair_id"]}')
                ov=r.get('overlap',[])
                if ov:
                    v=min(ov)
                    bucket=next((label for upper,label in [(.1,'below_.1'),(.3,'.1_to_.3'),(.5,'.3_to_.5'),(.7,'.5_to_.7')] if v<upper),'.7_and_above')
                    support[r['domain']+'/'+bucket]+=1
            manifests[subset]=dict(path=str(file),sha256=sha(file),pairs=len(data),by_domain=dict(domain_counts),
                pairs_per_scene=dict(scene_counts),geometry_support_fraction_buckets=dict(support))
        train,val=records['train'],records['val']
        def hashes(rows): return {r[k] for r in rows for k in ('source_sha256_a','source_sha256_b')}
        shared_hashes=hashes(train)&hashes(val)
        shared_groups={r['group'] for r in train}&{r['group'] for r in val}
        shared_ids={r['pair_id'] for r in train}&{r['pair_id'] for r in val}
        if shared_hashes: errors.append(f'Train/val shares {len(shared_hashes)} source byte hashes')
        if shared_groups: errors.append('Train/val shares groups: '+','.join(sorted(shared_groups)))
        if shared_ids: errors.append(f'Train/val shares {len(shared_ids)} pair IDs')
        for subset,rows in records.items():
            if {r['domain'] for r in rows}!={'megadepth','uav'}:
                errors.append(f'{subset} missing a real domain')
    else:
        warnings.append('Metadata-only check: train/val manifests and source byte hashes were not inspected')
    return dict(status='failed' if errors else 'metadata_and_manifest_pass' if manifest_checked else 'metadata_only_pass',
        preparation_file=str(preparation),preparation_sha256=sha(preparation),scene_inventory=counts,
        excluded_test_scene_ids=['0015','0022'],test_scene_ids_in_inventory=forbidden,
        scene_id_exclusion_passed=not forbidden,manifest_checked=manifest_checked,manifests=manifests,
        errors=errors,warnings=warnings,scope='Current geometry cache only; no audit of ancestor checkpoints or re-encoded duplicate images',
        support_fraction_note='Existing overlap is trusted geometry support fraction, not independently measured physical overlap',
        geometric_label_correctness_verified=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry-cache',type=Path)
    p.add_argument('--preparation',type=Path,help='Metadata-only input when no geometry cache is accessible')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    preparation=args.preparation or (args.geometry_cache/'preparation.json' if args.geometry_cache else None)
    if preparation is None: p.error('Supply --geometry-cache or --preparation')
    result=audit(preparation,args.geometry_cache)
    content=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(content,encoding='utf-8')
    print(content)
    if result['errors']: raise SystemExit(1)


if __name__=='__main__': main()
