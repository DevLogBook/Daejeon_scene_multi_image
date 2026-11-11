"""Shared contracts for statistics, train-only center augmentation and evaluation."""
import hashlib
import json
from pathlib import Path

import torch

from .adaptive import load_adaptive_state
from .config import GRMMatcherConfig
from .matcher import GRMMatcher
from .ordinal_search import body_fingerprint


ARMS={'natural':(1.,0.,0.),'synthetic':(0.,1.,0.),'mixed':(.5,.5,0.),'mixed_hn':(.4,.4,.2)}


def load_model(path,device='cpu',ordinal=False):
    raw=torch.load(path,map_location='cpu',weights_only=False)
    values=dict(raw['config'],use_amp=False,adaptive_variance=False)
    if ordinal:values.update(adaptive_enabled=True,adaptive_risk_kind='ordinal_local',adaptive_mode='learned_5_7_9')
    else:values.update(adaptive_enabled=False)
    config=GRMMatcherConfig.from_dict(values)
    if config.architecture_version!='v3' or config.mid_stride!=8:
        raise ValueError('This protocol requires V3 with actual F8 Mid features')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026);model=GRMMatcher(config)
    state={k:v for k,v in raw['model'].items() if not k.startswith('variance_calibrator.')}
    if not ordinal:state={k:v for k,v in state.items() if not k.startswith(('search_risk_head.','search_level_head.'))}
    load_adaptive_state(model,state,upgrade=ordinal)
    model.to(device).eval()
    return model,raw


def move(batch,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temporary.replace(path)


def code_fingerprint():
    h=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def verify_pool(report,info,model):
    if report.get('split')!='train':raise ValueError('An empirical reservoir must come from TRAIN, never val/test')
    if report.get('body_fingerprint')!=body_fingerprint(model):raise ValueError('Empirical offsets come from a different frozen matcher/config')
    for key in ('train_manifest_sha256','split_membership_sha256','kind','domain','size','coarse_stride'):
        if report['data'][key]!=info[key]:raise ValueError('Empirical offset source/split mismatch: '+key)
    return report['offset_reservoir_f8']


def source_arguments(parser):
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data-dir',type=Path)
    source.add_argument('--geometry-cache',type=Path)
    source.add_argument('--uav-prepared',type=Path)
    parser.add_argument('--uav-root',type=Path)
    parser.add_argument('--uav-split-file',type=Path)
    parser.add_argument('--image-cache',type=Path)
    parser.add_argument('--domain',choices=('uav','megadepth'),default='uav')


def source_from_args(args,config,split):
    if getattr(args,'uav_prepared',None):
        from .search_offset_data import PartitionedUAVSearchDataset
        if not args.uav_root or not args.uav_split_file or args.size!=512:
            raise ValueError('Repurposed UAV training needs --uav-root, --uav-split-file and size512')
        dataset=PartitionedUAVSearchDataset(args.uav_prepared,args.uav_root,config,args.uav_split_file,split,
            image_cache=getattr(args,'image_cache',None))
        return dataset,dataset.source_info
    from .search_offset_data import make_source
    return make_source(args.data_dir or args.geometry_cache,'homography' if args.data_dir else 'geometry',
                       split,config,args.size,args.domain)
