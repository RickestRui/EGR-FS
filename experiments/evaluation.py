"""One evaluation path for baseline, fine-tuning control and EGR-FS; raw counts to CSV."""
import time
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluate_graph import boundary_counts, small_fire_counts
from utils.tools import set_reproducibility
from utils.training_tools import get_dataset, get_model, Trainer
from experiments.protocol import write_csv, dump_json, sha


def cm_metrics(cm, names):
    cm = np.asarray(cm, dtype=np.float64)
    tp, actual, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    divide = lambda a,b: np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b>0)
    iou = divide(tp, actual+predicted-tp)
    p, r = divide(tp,predicted), divide(tp,actual)
    f1 = divide(2*tp, actual+predicted)
    result = dict(miou=float(iou.mean()), macro_precision=float(p.mean()), macro_recall=float(r.mean()),
                  macro_f1=float(f1.mean()), pixel_accuracy=float(tp.sum()/max(cm.sum(),1)),
                  frequency_weighted_f1=float((f1*actual).sum()/max(actual.sum(),1)))
    for i,name in enumerate(names):
        result.update({f'{name}_{key}':float(value[i]) for key,value in
                       [('iou',iou),('precision',p),('recall',r),('f1',f1),('tp',tp),
                        ('fp',predicted-tp),('fn',actual-tp)]})
    return result


def corrupt(images, condition):
    x = images.clone()
    if condition=='clean': return x
    if condition=='rgb_missing': x[:,:3] = 0
    elif condition=='ir_missing': x[:,3:] = 0
    elif condition.startswith('rgb_dark_'):
        x[:,:3] *= {'rgb_dark_05':.5, 'rgb_dark_025':.25}[condition]
    elif condition.startswith('ir_shift_'):
        shift = int(condition.rsplit('_',1)[1])
        x[:,3:] = F.pad(x[:,3:,:,:-shift], (shift,0,0,0))
    else: raise ValueError(f'Unknown condition {condition}')
    return x


@torch.inference_mode()
def evaluate(config, checkpoint, out, split='val', condition='clean', device=None,
             limit=None, warmup=20, repeats=100, save_predictions=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Evaluation already exists: {out}; select a different result root to rerun.')
    args = dict(config, PRETRAINED=None, CHECKPOINT=str(checkpoint), NUM_WORKERS=0)
    if device: args['DEVICE'] = device
    if not args.get('EVAL_RESIZE'):
        raise ValueError('Publication evaluator requires deterministic EVAL_RESIZE: true')
    if warmup < 1 or repeats < 1 or (limit is not None and limit < 1):
        raise ValueError('warmup/repeats/limit must be positive')
    set_reproducibility(args['SEED'])
    _, val, test = get_dataset(args, test=True)
    dataset = val if split=='val' else test
    full_size = len(dataset)
    sample_ids = list(dataset.files)
    if limit:
        dataset = Subset(dataset, range(min(limit,full_size)))
    if len(dataset)==0: raise ValueError('Empty evaluation split')
    model = get_model(args)
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    model.eval()
    # Teacher is training-only: remove its storage before reporting deployment cost.
    if hasattr(model,'teacher'): model.teacher = None
    dev = torch.device(args['DEVICE'])
    sync = lambda: torch.cuda.synchronize(dev) if dev.type=='cuda' else None
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    classes = args['NUM_CLASSES']; names = args['CLS_NAMES']
    channels = ('coarse','refined') if args['MODEL']=='robofire_graph' else ('prediction',)
    cms={c:np.zeros((classes,classes),np.int64) for c in channels}
    bounds={c:np.zeros(4,np.int64) for c in channels}
    small={c:{a:np.zeros(2,np.int64) for a in (16,64,256)} for c in channels}
    cal={c:np.zeros((15,3),float) for c in channels}
    nll={c:0. for c in channels}; brier=dict(nll)
    thresholds=np.linspace(.05,.95,19)
    pr={c:np.zeros((len(thresholds),3),np.int64) for c in channels}
    rows=[]; timing=[]; valid_total=0; repaired=damaged=0; example=None
    for index,batch in enumerate(loader):
        x=corrupt(batch[0].to(dev,dtype=torch.float32),condition)
        if example is None:
            example=x
            for _ in range(warmup): model(x)
            sync()
        gt=batch[1][0].numpy().astype(np.int64)
        valid=gt!=args['IGNORE_LABEL']
        sync(); start=time.perf_counter(); output=model(x); sync()
        timing.append((time.perf_counter()-start)*1000)
        logits_by_channel=({c:output[c+'_logits'] for c in channels} if len(channels)==2
                           else {'prediction':Trainer.segmentation_logits(output)})
        preds={}
        valid_total+=int(valid.sum())
        for channel,logits in logits_by_channel.items():
            logits=F.interpolate(logits,size=gt.shape,mode='bilinear',align_corners=args['ALIGN_CORNERS'])
            prob=logits.softmax(1)[0].cpu().numpy()
            pred=prob.argmax(0); preds[channel]=pred
            cm=np.bincount((gt[valid]*classes+pred[valid]).ravel(),minlength=classes**2).reshape(classes,classes)
            cms[channel]+=cm
            bc=boundary_counts(pred,gt,valid,tolerance=2); bounds[channel]+=bc
            row=dict(sample_id=sample_ids[index],output=channel,valid_pixels=int(valid.sum()),
                     fire_fraction=float(((gt==2)&valid).sum()/max(valid.sum(),1)))
            row.update(cm_metrics(cm,names))
            row.update({f'cm_{i}_{j}':int(cm[i,j]) for i in range(classes) for j in range(classes)})
            row.update({f'boundary_count_{i}':int(v) for i,v in enumerate(bc)})
            for area in small[channel]:
                sc=small_fire_counts(pred,gt,valid,max_area=area,match_iou=.1,fire_class=names.index('fire'))
                small[channel][area]+=sc
                row.update({f'small_fire_{area}_matched':int(sc[0]), f'small_fire_{area}_total':int(sc[1])})
            rows.append(row)
            pv=prob[:,valid].T; y=gt[valid]
            if len(y):
                confidence=pv.max(1); correct=(pv.argmax(1)==y)
                bins=np.minimum((confidence*15).astype(int),14)
                cal[channel][:,0]+=np.bincount(bins,minlength=15)
                cal[channel][:,1]+=np.bincount(bins,weights=confidence,minlength=15)
                cal[channel][:,2]+=np.bincount(bins,weights=correct,minlength=15)
                nll[channel]+=float(-np.log(pv[np.arange(len(y)),y].clip(1e-12)).sum())
                brier[channel]+=float(((pv-np.eye(classes)[y])**2).sum())
                true_fire=y==names.index('fire')
                for k,t in enumerate(thresholds):
                    pf=pv[:,names.index('fire')]>=t
                    pr[channel][k]+=[int((pf&true_fire).sum()),int((pf&~true_fire).sum()),int((~pf&true_fire).sum())]
            if save_predictions:
                from PIL import Image
                target=out/'predictions'/channel
                target.mkdir(parents=True,exist_ok=True)
                Image.fromarray(pred.astype(np.uint8)).save(target/f'{index:05d}.png')
        if len(channels)==2:
            before=preds['coarse']==gt; after=preds['refined']==gt
            repaired+=int((valid&~before&after).sum()); damaged+=int((valid&before&~after).sum())
    # Dedicated steady-state forward benchmark on one fixed input, excludes I/O/metrics.
    del output, logits_by_channel, logits
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
    speed=[]
    for _ in range(repeats):
        sync(); start=time.perf_counter(); model(example); sync()
        speed.append((time.perf_counter()-start)*1000)
    summaries=[]; matrices=[]; curves=[]; calibration=[]
    for c in channels:
        row=dict(experiment=args.get('EXPERIMENT_ID',args['MODEL']),seed=args['SEED'],dataset=args.get('DATASET_ID','FLAME2'),
                 split=split,condition=condition,output=c,samples=len(dataset),full_split=len(dataset)==full_size,
                 checkpoint_sha256=sha(checkpoint),**cm_metrics(cms[c],names))
        bp=bounds[c][0]/max(bounds[c][1],1); br=bounds[c][2]/max(bounds[c][3],1)
        row.update(boundary_precision=bp,boundary_recall=br,boundary_f1=2*bp*br/max(bp+br,1e-12),
                   nll=nll[c]/max(valid_total,1),brier=brier[c]/max(valid_total,1))
        bins=cal[c]
        row['ece15']=float(np.abs(bins[:,1]-bins[:,2]).sum()/max(valid_total,1))
        for a,counts in small[c].items():
            row[f'small_fire_recall_le{a}']=float(counts[0]/counts[1]) if counts[1] else ''
            row[f'small_fire_total_le{a}']=int(counts[1])
        row.update(repaired_pixels=repaired if c=='refined' else '', damaged_pixels=damaged if c=='refined' else '',
                   net_repaired_pixels=repaired-damaged if c=='refined' else '')
        summaries.append(row)
        matrices += [dict(output=c,true_class=names[i],predicted_class=names[j],count=int(cms[c][i,j]))
                     for i in range(classes) for j in range(classes)]
        for k,t in enumerate(thresholds):
            tp,fp,fn=map(int,pr[c][k]); precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
            curves.append(dict(output=c,threshold=float(t),tp=tp,fp=fp,fn=fn,
                               precision=precision,recall=recall,f1=2*tp/max(2*tp+fp+fn,1)))
        calibration += [dict(output=c,bin=k,count=int(v[0]),confidence_sum=v[1],correct_sum=v[2]) for k,v in enumerate(bins)]
    write_csv(out/'summary.csv',summaries)
    write_csv(out/'per_image.csv',rows)
    write_csv(out/'confusion.csv',matrices)
    write_csv(out/'fire_pr_curve.csv',curves)
    write_csv(out/'calibration.csv',calibration)
    write_csv(out/'latency_samples.csv',[dict(iteration=i,milliseconds=t) for i,t in enumerate(speed)])
    write_csv(out/'efficiency.csv',[dict(parameters=sum(p.numel() for p in model.parameters()),
               latency_mean_ms=np.mean(speed),latency_median_ms=np.median(speed),latency_p95_ms=np.percentile(speed,95),
               fps=1000/np.mean(speed),dataset_forward_mean_ms=np.mean(timing),
               peak_allocated_cuda_mb=torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else '',
               batch_size=1,precision='float32',warmup=warmup,repeats=repeats,
               concurrent_jobs=os.environ.get('EXPERIMENT_PARALLEL_JOBS','unspecified'),
               visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','all'),
               device=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU',
               scope='whole model forward including graph search; excluding teacher, I/O and metrics')])
    dump_json(out/'evaluation.json',dict(config=args,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
              split=split,condition=condition,limit=limit,selected_epoch=state.get('best_epoch'),
              torch_version=torch.__version__,cuda_version=torch.version.cuda,
              undefined_policy='zero denominator -> 0; missing GT small fires -> empty',
              boundary='4-neighbor transitions, dilation tolerance 2 evaluated pixels',
              small_fire='8-connected GT components, cumulative areas <=16/64/256, one-to-one IoU >=0.1 matching'))
    return summaries


@torch.inference_mode()
def benchmark(config, checkpoint, out, device='cuda:0', warmup=50, repeats=300):
    """Dedicated whole-model timing; no one-image accuracy reported."""
    out=Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    if warmup<1 or repeats<1: raise ValueError('Positive warmup and repeats required')
    args=dict(config,PRETRAINED=None,CHECKPOINT=str(checkpoint),DEVICE=device,NUM_WORKERS=0)
    set_reproducibility(args['SEED'])
    model=get_model(args)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model_state_dict'],strict=True)
    if hasattr(model,'teacher'): model.teacher=None
    model.eval(); del state
    _,val,_=get_dataset(args,test=True)
    example=torch.as_tensor(val[0][0]).unsqueeze(0).float().to(device)
    dev=torch.device(device)
    sync=lambda: torch.cuda.synchronize(dev) if dev.type=='cuda' else None
    for _ in range(warmup): model(example)
    sync()
    if dev.type=='cuda': torch.cuda.reset_peak_memory_stats(dev)
    durations=[]
    for _ in range(repeats):
        sync(); start=time.perf_counter(); model(example); sync()
        durations.append((time.perf_counter()-start)*1000)
    write_csv(out/'latency_samples.csv',[dict(iteration=i,milliseconds=t) for i,t in enumerate(durations)])
    write_csv(out/'efficiency.csv',[dict(experiment=args['EXPERIMENT_ID'],seed=args['SEED'],
        parameters=sum(p.numel() for p in model.parameters()),latency_mean_ms=np.mean(durations),
        latency_median_ms=np.median(durations),latency_p95_ms=np.percentile(durations,95),
        fps=1000/np.mean(durations),batch_size=1,precision='float32',warmup=warmup,repeats=repeats,
        peak_allocated_cuda_mb=torch.cuda.max_memory_allocated(dev)/2**20 if dev.type=='cuda' else '',
        device=torch.cuda.get_device_name(dev) if dev.type=='cuda' else 'CPU',
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','all'),
        scope='complete forward including graph search; no teacher, I/O or metrics',
        reference_sample=val.files[0],checkpoint_sha256=sha(checkpoint))])
    dump_json(out/'completed.json',dict(config=config,checkpoint_sha256=sha(checkpoint),
              torch_version=torch.__version__,cuda_version=torch.version.cuda,warmup=warmup,repeats=repeats))
