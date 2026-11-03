"""CLI: python -m experiments.paper --help (run from project root)."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from experiments.protocol import (SEEDS, VARIANTS, EXPERIMENT_CHOICES, LEGACY_METHOD_ID, canonical_experiment, CONDITIONS, prepare, audit, sha,
                                  source_hashes, write_csv, read_csv, dump_json, command_table)


def config_for(root, name, seed):
    path = root/'configs'/f'{name}_s{seed}.yaml'
    if not path.exists() and canonical_experiment(name) == 'egr_fs':
        alternate = LEGACY_METHOD_ID if name == 'egr_fs' else 'egr_fs'
        path = root/'configs'/f'{alternate}_s{seed}.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8')), path


def checkpoint_for(args):
    return Path('weights')/args['PROJECTNAME']/args['SESSIONAME']/'best.pth'


def checked_audit(root, args):
    path=root/'audit'/'audit.json'
    if not path.exists(): raise ValueError('Run audit before training or evaluation.')
    info=json.loads(path.read_text(encoding='utf-8'))
    if not info['ok']: raise ValueError('Dataset audit failed; inspect audit/audit.json')
    for split,key in [('train','TRAINSET'),('val','VALIDSET'),('test','TESTSET')]:
        if sha(Path(args['ROOTDATASET'])/args[key])!=info['splits'][split]['sha256']:
            raise ValueError('Split changed after audit. Create a new protocol for a changed split.')
    for row in read_csv(root/'audit'/'audit_samples.csv'):
        for modality in ('rgb','ir','gt'):
            path=Path(args['ROOTDATASET'])/row['sample_id'].replace('XXX',modality)
            if sha(path)!=row[f'{modality}_sha256']:
                raise ValueError(f'Dataset content changed after audit: {path}')
    return info


def train_run(root, name, seed, device, resume):
    from train import main
    args,path=config_for(root,name,seed)
    audit_info=checked_audit(root,args)
    if device: args['DEVICE']=device
    session=Path('outputs')/args['PROJECTNAME']/args['SESSIONAME']
    session.mkdir(parents=True,exist_ok=True)
    if resume:
        import torch
        original=json.loads((session/'run.json').read_text(encoding='utf-8'))
        if original['source_hashes']!=source_hashes():
            raise ValueError('Source changed since training began; use a new experiment protocol.')
        checkpoint=checkpoint_for(args).with_name('latest.pth')
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        previous=torch.load(checkpoint,map_location='cpu',weights_only=False)['training_config']
        for k,v in args.items():
            if k not in ('DEVICE','NUM_WORKERS','CHECKPOINT') and previous.get(k)!=v:
                raise ValueError(f'Resume config changed: {k}')
        args['CHECKPOINT']=str(checkpoint)
    if not args['CHECKPOINT'] and not Path(args['PRETRAINED']).is_file():
        raise FileNotFoundError(f'Initializer missing: {args["PRETRAINED"]}. Train the seed-matched RFFNet first.')
    # Record effective training config and source identity before optimizer updates.
    run=dict(config=args,config_sha256=sha(path),source_hashes=source_hashes(),
             initializer_sha256=sha(args['PRETRAINED']) if Path(args['PRETRAINED']).exists() else None,
             audit_fingerprint=audit_info['fingerprint'])
    if (session/'run.json').exists() and not resume:
        raise FileExistsError(f'Run exists: {session}')
    dump_json(session/('resume.json' if resume else 'run.json'),run)
    start=time.perf_counter(); status='failed'
    try:
        main(args); status='completed'
    finally:
        filename='resume_runtime.csv' if resume else 'training_runtime.csv'
        write_csv(session/filename,[dict(experiment=name,seed=seed,seconds=time.perf_counter()-start,status=status)])


def lock(root, names, seeds):
    lock_path=root/'test_lock.json'
    if lock_path.exists(): raise FileExistsError('Test lock already exists; preserve it and use a new protocol for changes.')
    records=[]
    for name in names:
        for seed in seeds:
            args,path=config_for(root,name,seed)
            info=checked_audit(root,args)
            checkpoint=checkpoint_for(args)
            if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
            records.append(dict(experiment=name,seed=seed,checkpoint=str(checkpoint),
                                checkpoint_sha256=sha(checkpoint),config_sha256=sha(path),
                                audit_fingerprint=info['fingerprint']))
    dump_json(lock_path,dict(selection_rule='validation mIoU; locked before final test',
                            records=records,source_hashes=source_hashes()))


def run_evaluation(root, name, seed, options):
    import torch
    from experiments.evaluation import evaluate
    args,path=config_for(root,name,seed)
    info=checked_audit(root,args)
    checkpoint=checkpoint_for(args)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    trained=state['training_config']
    # Use the settings that actually produced the checkpoint, never a new ablation at test time.
    for k,v in args.items():
        if k not in ('DEVICE','NUM_WORKERS','CHECKPOINT') and trained.get(k)!=v:
            raise ValueError(f'Config differs from checkpoint: {k}')
    del state
    if options.split=='test':
        sealed=json.loads((root/'test_lock.json').read_text(encoding='utf-8'))
        found=[r for r in sealed['records'] if r['experiment']==name and r['seed']==seed]
        if len(found)!=1 or found[0]['checkpoint_sha256']!=sha(checkpoint) or found[0]['config_sha256']!=sha(path):
            raise ValueError('Test checkpoint/config is not the locked validation selection')
        if found[0]['audit_fingerprint']!=info['fingerprint'] or sealed['source_hashes']!=source_hashes():
            raise ValueError('Data audit or source changed after test lock')
    for condition in options.conditions:
        leaf=f'{options.split}_{condition}' + (f'_smoke{options.limit}' if options.limit else '')
        output=root/'results'/f'{name}_s{seed}'/leaf
        print(evaluate(trained,checkpoint,output,options.split,condition,options.device,
                       options.limit,options.warmup,options.repeats,options.save_predictions))


def aggregate(root):
    records=[]; efficiency=[]
    for path in sorted((root/'results').glob('*/*/summary.csv')):
        complete=[]
        for row in read_csv(path):
            if row['full_split']!='True' or '_smoke' in str(path): continue
            records.append(dict(row,experiment=canonical_experiment(row['experiment']),source=str(path)))
            complete.append(row)
        if complete and (path.parent/'efficiency.csv').exists():
            metadata={k:complete[0][k] for k in ('experiment','seed','dataset','split','condition')}
            efficiency.extend(dict(metadata,**r) for r in read_csv(path.parent/'efficiency.csv'))
    write_csv(root/'all_results.csv',records, None if records else ['experiment','seed','split','condition','output','miou'])
    write_csv(root/'efficiency_results.csv',efficiency,
              None if efficiency else ['experiment','seed','split','condition','parameters','latency_mean_ms','fps'])
    benchmarks=[]
    for path in sorted((root/'benchmarks').glob('*/efficiency.csv')):
        if (path.parent/'completed.json').exists(): benchmarks.extend(read_csv(path))
    write_csv(root/'benchmark_results.csv',benchmarks,
              None if benchmarks else ['experiment','seed','parameters','latency_mean_ms','fps'])
    groups={}
    for row in records:
        key=tuple(row[k] for k in ('experiment','dataset','split','condition','output'))
        groups.setdefault(key,[]).append(row)
    results=[]
    metrics=['miou','macro_f1','macro_precision','macro_recall','pixel_accuracy','fire_iou','smoke_iou',
             'background_iou','fire_precision','fire_recall','boundary_f1','small_fire_recall_le16',
             'small_fire_recall_le64','small_fire_recall_le256','ece15','nll','brier']
    for key,rows in groups.items():
        if len({r['seed'] for r in rows})!=len(rows): raise ValueError('Duplicate seed results in aggregation')
        for metric in metrics:
            values=[float(r[metric]) for r in rows if r.get(metric,'')!='']
            results.append(dict(zip(('experiment','dataset','split','condition','output'),key),
                                metric=metric,n=len(values),seeds=';'.join(r['seed'] for r in rows),
                                mean=np.mean(values) if values else '',
                                std=np.std(values,ddof=1) if len(values)>1 else ''))
    write_csv(root/'seed_summary.csv',results, None if results else ['experiment','metric','n','mean','std'])
    paired=[]
    for comparator in ('rffnet','finetune'):
        for split in ('val','test'):
            for condition in CONDITIONS:
                left={r['seed']:r for r in records if r['experiment']==comparator and r['split']==split and
                      r['condition']==condition and r['output']==('prediction' if comparator=='rffnet' else 'refined')}
                right={r['seed']:r for r in records if r['experiment']=='egr_fs' and r['split']==split and
                       r['condition']==condition and r['output']=='refined'}
                seeds=sorted(left.keys()&right.keys())
                if not seeds: continue
                for metric in metrics:
                    diff=[float(right[s][metric])-float(left[s][metric]) for s in seeds
                          if left[s].get(metric,'')!='' and right[s].get(metric,'')!='']
                    if diff:
                        paired.append(dict(comparison='egr_fs_minus_'+comparator,split=split,condition=condition,
                                           metric=metric,n=len(diff),mean_delta=np.mean(diff),
                                           std_delta=np.std(diff,ddof=1) if len(diff)>1 else ''))
    write_csv(root/'paired_seed_deltas.csv',paired, None if paired else ['comparison','metric','n','mean_delta','std_delta'])
    return len(records)


def bootstrap(left, right, out, left_output, right_output, groups=None, draws=2000):
    from experiments.evaluation import cm_metrics
    a={r['sample_id']:r for r in read_csv(left) if r['output']==left_output}
    b={r['sample_id']:r for r in read_csv(right) if r['output']==right_output}
    if not a or a.keys()!=b.keys(): raise ValueError('Paired comparison requires identical nonempty sample IDs')
    mapping={r['sample_id']:r['group_id'] for r in read_csv(groups)} if groups else {s:s for s in a}
    if any(not mapping.get(s) for s in a): raise ValueError('Missing sequence group')
    blocks=sorted({mapping[s] for s in a})
    sums={k:np.zeros((len(blocks),3,3)) for k in ('a','b')}; idx={k:i for i,k in enumerate(blocks)}
    for sample in a:
        for key,rows in [('a',a),('b',b)]:
            sums[key][idx[mapping[sample]]]+=np.array([float(rows[sample][f'cm_{i}_{j}'])
                                                       for i in range(3) for j in range(3)]).reshape(3,3)
    rng=np.random.default_rng(2026); metrics=('miou','fire_iou','smoke_iou','fire_precision','fire_recall')
    deltas={m:[] for m in metrics}
    for _ in range(draws):
        sampled=rng.integers(0,len(blocks),len(blocks))
        ma=cm_metrics(sums['a'][sampled].sum(0),['background','smoke','fire'])
        mb=cm_metrics(sums['b'][sampled].sum(0),['background','smoke','fire'])
        for m in metrics: deltas[m].append(mb[m]-ma[m])
    ma=cm_metrics(sums['a'].sum(0),['background','smoke','fire'])
    mb=cm_metrics(sums['b'].sum(0),['background','smoke','fire'])
    write_csv(out,[dict(metric=m,delta=mb[m]-ma[m],ci025=np.quantile(deltas[m],.025),
                        ci975=np.quantile(deltas[m],.975),draws=draws,blocks=len(blocks),
                        unit='sequence' if groups else 'frame_descriptive_only') for m in metrics])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('experiments/egr_fs_suite'))
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare'); prep.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    prep.add_argument('--base-epochs',type=int,default=500); prep.add_argument('--graph-epochs',type=int,default=100)
    au=sub.add_parser('audit'); au.add_argument('--groups',type=Path)
    tr=sub.add_parser('train'); tr.add_argument('--experiment',required=True,choices=EXPERIMENT_CHOICES)
    tr.add_argument('--seed',type=int,default=200); tr.add_argument('--device'); tr.add_argument('--resume',action='store_true')
    lk=sub.add_parser('lock'); lk.add_argument('--experiments',nargs='+',required=True,choices=EXPERIMENT_CHOICES)
    lk.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    ev=sub.add_parser('evaluate'); ev.add_argument('--experiment',required=True,choices=EXPERIMENT_CHOICES)
    ev.add_argument('--seed',type=int,default=200); ev.add_argument('--split',choices=['val','test'],default='val')
    ev.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=['clean'])
    ev.add_argument('--device'); ev.add_argument('--limit',type=int)
    ev.add_argument('--warmup',type=int,default=20); ev.add_argument('--repeats',type=int,default=100)
    ev.add_argument('--save-predictions',action='store_true')
    sub.add_parser('aggregate')
    sub.add_parser('commands')
    bm=sub.add_parser('benchmark'); bm.add_argument('--experiment',required=True,choices=EXPERIMENT_CHOICES)
    bm.add_argument('--seed',type=int,default=200); bm.add_argument('--device',default='cuda:0')
    bm.add_argument('--warmup',type=int,default=50); bm.add_argument('--repeats',type=int,default=300)
    boot=sub.add_parser('bootstrap'); boot.add_argument('--left',required=True); boot.add_argument('--right',required=True)
    boot.add_argument('--output',required=True); boot.add_argument('--left-output',default='prediction')
    boot.add_argument('--right-output',default='refined'); boot.add_argument('--groups'); boot.add_argument('--draws',type=int,default=2000)
    o=p.parse_args(); root=o.root
    if root.is_absolute() or '..' in root.parts: p.error('--root must be a project-relative path without ..')
    if o.command=='prepare': print('Prepared',len(prepare(root,o.seeds,o.base_epochs,o.graph_epochs)),'runs')
    elif o.command=='audit':
        first=read_csv(root/'manifest.csv')[0]
        args=yaml.safe_load(Path(first['config']).read_text(encoding='utf-8'))
        report=audit(args,root/'audit',o.groups); print(json.dumps(report,ensure_ascii=False,indent=2))
        if not report['ok']: raise SystemExit(1)
    elif o.command=='train': train_run(root,o.experiment,o.seed,o.device,o.resume)
    elif o.command=='lock': lock(root,o.experiments,o.seeds)
    elif o.command=='evaluate': run_evaluation(root,o.experiment,o.seed,o)
    elif o.command=='aggregate': print('Aggregated',aggregate(root),'full-split result rows')
    elif o.command=='commands': command_table(root)
    elif o.command=='benchmark':
        import torch
        from experiments.evaluation import benchmark
        args,path=config_for(root,o.experiment,o.seed); checked_audit(root,args)
        checkpoint=checkpoint_for(args)
        trained=torch.load(checkpoint,map_location='cpu',weights_only=False)['training_config']
        for k,v in args.items():
            if k not in ('DEVICE','NUM_WORKERS','CHECKPOINT') and trained.get(k)!=v:
                raise ValueError(f'Config differs from checkpoint: {k}')
        benchmark(trained,checkpoint,root/'benchmarks'/f'{o.experiment}_s{o.seed}',o.device,o.warmup,o.repeats)
    elif o.command=='bootstrap':
        if o.draws<100: p.error('Use at least 100 bootstrap draws')
        bootstrap(o.left,o.right,o.output,o.left_output,o.right_output,o.groups,o.draws)


if __name__=='__main__': main()
