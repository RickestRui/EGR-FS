"""Experiment definitions, provenance, CSV and split auditing (no GPU needed)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SEEDS = (200, 201, 202)
# Every change is relative to the full EGR-FS configuration, not a new main method.
VARIANTS = {
    'rffnet': ('main', 'Stage-one RFFNet baseline', {}),
    'finetune': ('main', 'Same stage-two scope/loss/teacher, no refiner or node loss',
                 {'REFINER_KIND': 'none', 'GRAPH_NODE_WEIGHT': 0.}),
    'egr_fs': ('main', 'Full EGR-FS', {}),
    'self_only': ('ablation', 'Self edges only; same trainable graph operators',
                  {'GRAPH_SPATIAL': False, 'GRAPH_NONLOCAL_K': 0}),
    'local_only': ('ablation', 'Remove nonlocal edges', {'GRAPH_NONLOCAL_K': 0}),
    'no_evidence': ('ablation', 'Zero E in nodes, neighbor descriptors, edges, prior and adapter',
                    {'GRAPH_USE_EVIDENCE': False}),
    'no_prior': ('ablation', 'Remove evidence log prior only', {'GRAPH_USE_PRIOR': False}),
    'no_edge_attr': ('ablation', 'Zero edge attributes, retain topology and soft prior',
                     {'GRAPH_USE_EDGE_ATTR': False}),
    'no_predictions': ('ablation', 'Zero detached P, entropy, boundary cues in nodes/edges',
                       {'GRAPH_USE_PREDICTIONS': False}),
    'no_node_loss': ('ablation', 'Remove regional soft-target loss', {'GRAPH_NODE_WEIGHT': 0.}),
    'no_teacher': ('ablation', 'Remove both teacher-derived loss terms',
                    {'USE_TEACHER': False, 'TEACHER_WEIGHT': 0., 'TEACHER_ERROR_WEIGHT': 0.}),
    'teacher_all': ('ablation', 'KL on all labeled pixels; retain error CE',
                     {'TEACHER_SELECTION': 'all'}),
    'random_adapter': ('ablation', 'Standard output-convolution initialization',
                        {'ZERO_INIT_ADAPTER': False}),
    'frozen': ('ablation', 'Freeze complete backbone', {'BACKBONE_MODE': 'frozen'}),
    'k2': ('sensitivity', 'Nonlocal k=2', {'GRAPH_NONLOCAL_K': 2}),
    'k8': ('sensitivity', 'Nonlocal k=8', {'GRAPH_NONLOCAL_K': 8}),
    'patch4': ('sensitivity', 'Grid stride 4 (higher neighbor-search cost)', {'GRAPH_PATCH_SIZE': 4}),
    'patch16': ('sensitivity', 'Grid stride 16', {'GRAPH_PATCH_SIZE': 16}),
    'layer1': ('sensitivity', 'One attention layer', {'GRAPH_LAYERS': 1}),
    'layer3': ('sensitivity', 'Three attention layers', {'GRAPH_LAYERS': 3}),
    'teacher01': ('sensitivity', 'Retention weight 0.1', {'TEACHER_WEIGHT': .1}),
    'teacher05': ('sensitivity', 'Retention weight 0.5', {'TEACHER_WEIGHT': .5}),
}

# Legacy identifiers are accepted only for reading completed experiment protocols.
LEGACY_METHOD_ID = 'v4'
EXPERIMENT_CHOICES = tuple(VARIANTS) + (LEGACY_METHOD_ID,)


def canonical_experiment(name):
    return 'egr_fs' if name == LEGACY_METHOD_ID else name

CONDITIONS = ('clean', 'rgb_dark_05', 'rgb_dark_025', 'rgb_missing', 'ir_missing',
              'ir_shift_2', 'ir_shift_4', 'ir_shift_8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def dump_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def source_hashes():
    files = [Path('train.py'), Path('evaluate_graph.py')] + sorted(Path('models').glob('*.py')) + sorted(Path('utils').glob('*.py'))
    files += sorted(Path('datasets').glob('*.py')) + sorted(Path('experiments').glob('*.py'))
    return {p.as_posix(): sha(p) for p in files}


def prepare(root, seeds=SEEDS, base_epochs=500, graph_epochs=100):
    root = Path(root)
    if (root/'manifest.csv').exists():
        raise FileExistsError('Protocol already exists; reuse it or choose a new --root. No files overwritten.')
    base = yaml.safe_load(Path('config/wildfire.yaml').read_text(encoding='utf-8'))
    graph = yaml.safe_load(Path('config/wildfire_graph.yaml').read_text(encoding='utf-8'))
    common = dict(PROJECTNAME=root.as_posix(), ONLINELOG=False, NUM_WORKERS=0,
                  EVAL_RESIZE=True, SAVE_VISUALIZATIONS=False, LABEL_ENCODING='palette',
                  VALID_FREQ=1, SAVE_EVERY=0, SAVE_BASELINE_CHECKPOINT=False,
                  SAVE_LATEST=True, EARLY_STOP_MIN_DELTA=0.,
                  CHECKPOINT=None, DATASET_ID='FLAME2', TEACHER_SELECTION='correct')
    switches = dict(REFINER_KIND='graph', GRAPH_USE_EVIDENCE=True, GRAPH_USE_PREDICTIONS=True,
                    GRAPH_USE_PRIOR=True, GRAPH_USE_EDGE_ATTR=True, GRAPH_SPATIAL=True,
                    ZERO_INIT_ADAPTER=True)
    rows = []
    for name, (family, question, changes) in VARIANTS.items():
        for seed in seeds:
            # One-seed validation sensitivity; confirm chosen setting with all main seeds.
            if family == 'sensitivity' and seed != seeds[0]:
                continue
            session = f'{name}_s{seed}'
            args = dict(base if name == 'rffnet' else graph)
            args.update(common, SEED=seed, SESSIONAME=session, EXPERIMENT_ID=name, **switches)
            if name == 'rffnet':
                args.update(EPOCHS=base_epochs, WARMUP=min(20, base_epochs-1), STOPCOUNTER=50)
            else:
                args.update(EPOCHS=graph_epochs, WARMUP=min(3, graph_epochs-1),
                            PRETRAINED=(Path('weights')/root/f'rffnet_s{seed}'/'best.pth').as_posix())
            args.update(changes)
            config = root/'configs'/f'{session}.yaml'
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(yaml.safe_dump(args, allow_unicode=True, sort_keys=False), encoding='utf-8')
            rows.append(dict(experiment=name, family=family, seed=seed, question=question,
                             config=config.as_posix(), checkpoint=(Path('weights')/root/session/'best.pth').as_posix(),
                             dependency='' if name=='rffnet' else f'rffnet_s{seed}', status='planned'))
    write_csv(root/'manifest.csv', rows)
    dump_json(root/'protocol.json', dict(dataset='FLAME2', seeds=list(seeds),
              primary_metric='global confusion matrix mIoU; background included',
              selection='validation mIoU only', size=[256,256], conditions=CONDITIONS,
              test_rule='lock selected checkpoint hashes before test',
              pretrained='Existing 9-class initializer; verify its provenance before publication, do not call it ImageNet-only.',
              source_hashes=source_hashes()))
    command_table(root)
    return rows


def command_table(root):
    root=Path(root)
    rows=[]
    for item in read_csv(root/'manifest.csv'):
        suffix=f"--experiment {item['experiment']} --seed {item['seed']}"
        prefix=f'python -m experiments.paper --root {root.as_posix()}'
        rows.append(dict(item,train_command=f'{prefix} train {suffix}',
                         validation_command=f'{prefix} evaluate {suffix} --split val',
                         test_command=f'{prefix} evaluate {suffix} --split test --save-predictions'
                         if item['family']!='sensitivity' else '',
                         test_requirement='Lock checkpoint before test; sensitivity stays validation-only'))
    write_csv(root/'commands.csv',rows)


def audit(config, out, groups=None):
    """Check paths, triplet shape, palette, duplicate RGB+IR bytes, split leakage."""
    out = Path(out)
    root = Path(config['ROOTDATASET'])
    group_map = {}
    if groups:
        for row in read_csv(groups):
            if row['sample_id'] in group_map:
                raise ValueError('Duplicate sample_id in groups CSV')
            group_map[row['sample_id']] = row['group_id'].strip()
    rows, problems, split_meta = [], [], {}
    seen_path, seen_pair, seen_group = {}, {}, {}
    for split, key in [('train','TRAINSET'), ('val','VALIDSET'), ('test','TESTSET')]:
        path = root/config[key]
        lines = [s.strip() for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        split_meta[split] = dict(path=path.as_posix(), sha256=sha(path), samples=len(lines))
        if not lines:
            problems.append(f'{split}: empty split')
        for sample in lines:
            files = [root/sample.replace('XXX', modality) for modality in ('rgb','ir','gt')]
            if sample in seen_path:
                problems.append(f'duplicate sample {sample}: {seen_path[sample]} and {split}')
            seen_path[sample] = split
            if not all(f.is_file() for f in files):
                problems.append(f'missing triplet: {sample}')
                continue
            hashes = [sha(f) for f in files]
            pair = tuple(hashes[:2])
            if pair in seen_pair:
                problems.append(f'duplicate RGB+IR bytes: {sample} vs {seen_pair[pair]}')
            seen_pair[pair] = (split,sample)
            with Image.open(files[0]) as im: rgb_size = im.size
            with Image.open(files[1]) as im: ir_size = im.size
            with Image.open(files[2]) as im: mask = np.asarray(im.convert('RGB'))
            if rgb_size != ir_size or rgb_size != (mask.shape[1], mask.shape[0]):
                problems.append(f'triplet size mismatch: {sample}')
            labels = np.full(mask.shape[:2], 255, dtype=np.uint8)
            for k, value in enumerate((0,125,255)):
                labels[(mask == value).all(-1)] = k
            if (labels == 255).any():
                problems.append(f'unrecognized palette pixels: {sample}')
            group = group_map.get(sample, '')
            if groups and not group:
                problems.append(f'missing sequence group: {sample}')
            if group and group in seen_group and seen_group[group] != split:
                problems.append(f'sequence group crosses splits: {group}')
            if group: seen_group[group] = split
            rows.append(dict(split=split, sample_id=sample, group_id=group,
                             width=rgb_size[0], height=rgb_size[1], rgb_sha256=hashes[0],
                             ir_sha256=hashes[1], gt_sha256=hashes[2],
                             background_pixels=int((labels==0).sum()), smoke_pixels=int((labels==1).sum()),
                             fire_pixels=int((labels==2).sum())))
    write_csv(out/'audit_samples.csv', rows)
    if not (out/'sequence_groups_template.csv').exists():
        write_csv(out/'sequence_groups_template.csv',
                  [dict(sample_id=r['sample_id'],group_id='') for r in rows], ['sample_id','group_id'])
    # Hash the sorted file content inventory as well as the list files.
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    report = dict(ok=not problems, problems=problems, splits=split_meta, fingerprint=fingerprint,
                  sequence_groups_checked=bool(groups),
                  note='Exact-file audit cannot establish absence of near-duplicate video frames or unseen-event generalization.')
    dump_json(out/'audit.json', report)
    return report
