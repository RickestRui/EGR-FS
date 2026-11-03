"""Independent one-GPU jobs with bounded parallelism, dependency barriers and logs.

Run from the project root; CUDA_VISIBLE_DEVICES maps each selected physical GPU
to cuda:0 inside its child. This launcher is not DDP.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from experiments.protocol import read_csv,write_csv,dump_json,sha,source_hashes,CONDITIONS
from experiments.paper import config_for,checkpoint_for


def select_rows(root,family):
    rows=read_csv(root/'manifest.csv')
    return [r for r in rows if family=='all' or r['family']==family or
            (family=='confirmatory' and r['family']!='sensitivity')]


def train_complete(root,row):
    args,config=config_for(root,row['experiment'],row['seed'])
    session=Path('outputs')/args['PROJECTNAME']/args['SESSIONAME']
    times=[p for p in (session/'training_runtime.csv',session/'resume_runtime.csv') if p.exists()]
    if not times or read_csv(max(times,key=lambda p:p.stat().st_mtime))[-1]['status']!='completed': return False
    if not checkpoint_for(args).exists(): raise ValueError(f'Completed run missing checkpoint: {session}')
    meta=json.loads((session/'run.json').read_text(encoding='utf-8'))
    if meta['config_sha256']!=sha(config) or meta['source_hashes']!=source_hashes():
        raise ValueError(f'Completed run provenance changed: {session}')
    if meta['initializer_sha256']!=sha(args['PRETRAINED']):
        raise ValueError(f'Initializer changed: {session}')
    return True


def jobs_for(root,rows,phase):
    jobs=[]
    for row in rows:
        conditions=CONDITIONS[1:] if phase=='robustness' else ['clean']
        for condition in conditions:
            experiment=row['experiment']; seed=row['seed']
            session=f'{experiment}_s{seed}'
            args=[sys.executable,'-u','-m','experiments.paper','--root',root.as_posix()]
            select=['--experiment',experiment,'--seed',str(seed)]
            if phase=='train': args+=['train']+select+['--device','cuda:0']
            elif phase=='benchmark': args+=['benchmark']+select+['--device','cuda:0']
            else:
                split='val' if phase=='val' else 'test'
                args+=['evaluate']+select+['--split',split,'--conditions',condition,'--device','cuda:0']
                if phase=='test': args+=['--save-predictions']
            jobs.append(dict(row=row,phase=phase,condition=condition,
                             id=f'{session}_{phase}_{condition}',command=args,status='pending',gpu=''))
    return jobs


def done(root,job):
    row=job['row']; phase=job['phase']
    complete=train_complete(root,row)
    if phase=='train': return complete
    if not complete: raise ValueError(f'Training incomplete: {row["experiment"]}_s{row["seed"]}')
    session=f"{row['experiment']}_s{row['seed']}"
    target=(root/'benchmarks'/session if phase=='benchmark' else
            root/'results'/session/f"{'val' if phase=='val' else 'test'}_{job['condition']}")
    marker=target/('completed.json' if phase=='benchmark' else 'evaluation.json')
    if not target.exists() or not any(target.iterdir()): return False
    if not marker.exists(): raise ValueError(f'Partial result requires inspection and archival before retry: {target}')
    meta=json.loads(marker.read_text(encoding='utf-8'))
    cfg,_=config_for(root,row['experiment'],row['seed'])
    if meta['checkpoint_sha256']!=sha(checkpoint_for(cfg)):
        raise ValueError(f'Checkpoint changed after evaluation: {target}')
    if phase!='benchmark':
        results=read_csv(target/'summary.csv')
        if not results or any(r['full_split']!='True' for r in results): raise ValueError(f'Incomplete split: {target}')
    return True


def persist(path,jobs):
    write_csv(path,[dict(job=j['id'],experiment=j['row']['experiment'],seed=j['row']['seed'],
                        phase=j['phase'],condition=j['condition'],gpu=j['gpu'],status=j['status'],
                        pid=j.get('pid',''),seconds=j.get('seconds',''),exit_code=j.get('exit_code',''),
                        log=j.get('log',''),command=' '.join(j['command'])) for j in jobs])


def run_jobs(root,jobs,gpus,resume,ledger):
    active={}; failed=False
    try:
        while True:
            for gpu,task in list(active.items()):
                proc,job,stream,start=task
                code=proc.poll()
                if code is None: continue
                stream.close(); job.update(exit_code=code,seconds=time.monotonic()-start,
                                           status='completed' if code==0 else 'failed')
                if code: failed=True
                del active[gpu]; persist(ledger,jobs)
                print(job['status'],job['id'],flush=True)
            # Do not dispatch dependents after any failure; active independent jobs may finish.
            for gpu in gpus:
                if failed or gpu in active: continue
                pending=[j for j in jobs if j['status']=='pending']
                if not pending: break
                job=pending[0]
                if done(root,job):
                    job['status']='skipped_completed'; persist(ledger,jobs)
                    continue
                if job['phase']=='train':
                    row=job['row']; cfg,_=config_for(root,row['experiment'],row['seed'])
                    if row['dependency'] and not train_complete(root,dict(experiment='rffnet',seed=row['seed'])):
                        raise ValueError(f'Complete stage one first: {row["dependency"]}')
                    latest=checkpoint_for(cfg).with_name('latest.pth')
                    run=Path('outputs')/cfg['PROJECTNAME']/cfg['SESSIONAME']/'run.json'
                    if latest.exists():
                        if not resume: raise ValueError(f'Interrupted run: rerun with --resume ({latest})')
                        job['command']+=['--resume']
                    elif run.exists():
                        raise ValueError(f'Run failed before first latest checkpoint; archive its outputs/weights before retry: {run.parent}')
                log=root/'queue_logs'/f"{job['id']}.log"; log.parent.mkdir(parents=True,exist_ok=True)
                stream=log.open('a',encoding='utf-8')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
                         EXPERIMENT_PARALLEL_JOBS=str(len(gpus)))
                proc=subprocess.Popen(job['command'],stdout=stream,stderr=subprocess.STDOUT,env=env)
                job.update(status='running',gpu=gpu,pid=proc.pid,log=log.as_posix())
                active[gpu]=(proc,job,stream,time.monotonic());persist(ledger,jobs)
                print('started',job['id'],'physical GPU',gpu,flush=True)
            if not active and (failed or not any(j['status']=='pending' for j in jobs)): break
            time.sleep(1)
    finally:
        for proc,job,stream,start in active.values():
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
            stream.close();job.update(status='interrupted',seconds=time.monotonic()-start,exit_code=proc.returncode)
        persist(ledger,jobs)
    if failed: raise RuntimeError(f'Queue stopped after failure; inspect {ledger}')


def preflight(root,gpus):
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    inventory=[]
    for index in gpus:
        gpu=int(index)
        if gpu>=torch.cuda.device_count(): raise ValueError(f'GPU {gpu} not visible')
        p=torch.cuda.get_device_properties(gpu)
        inventory.append(dict(index=gpu,name=p.name,memory_gib=p.total_memory/2**30,
                              torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,
                              project_directory=str(Path.cwd())))
    write_csv(root/'server_environment.csv',inventory)
    print(json.dumps(inventory,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('experiments/egr_fs_4090'))
    p.add_argument('--phase',required=True,choices=['preflight','train','val','test','robustness','benchmark'])
    p.add_argument('--family',default='main',choices=['main','ablation','sensitivity','confirmatory','all'])
    p.add_argument('--gpus',nargs='+',default=['0','1'])
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    o=p.parse_args()
    if o.root.is_absolute() or '..' in o.root.parts: p.error('root must be project-relative')
    if not all(s.isdigit() for s in o.gpus) or len(set(o.gpus))!=len(o.gpus): p.error('Distinct numeric GPU IDs required')
    if not Path('train.py').is_file(): p.error('Run from the EGR-FS project root')
    if o.phase=='preflight': preflight(o.root,o.gpus); return
    rows=select_rows(o.root,o.family)
    if not rows: p.error('No selected experiments')
    if o.phase in ('test','robustness') and any(r['family']=='sensitivity' for r in rows):
        p.error('Sensitivity is validation-only')
    if o.phase=='benchmark' and len(o.gpus)!=1: p.error('Use one GPU for isolated sequential timing')
    jobs=jobs_for(o.root,rows,o.phase)
    if o.dry_run:
        target=o.root/f'queue_plan_{o.phase}_{o.family}.csv';persist(target,jobs);print(target);return
    preflight(o.root,o.gpus)
    guard=o.root/'queue.lock'
    try: fd=os.open(guard,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: p.error(f'Queue lock exists: {guard}. Check its PID before archiving a stale lock.')
    os.write(fd,f'{os.getpid()}\n'.encode());os.close(fd)
    def stop(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        if o.phase=='train':
            # Finish every selected baseline before launching its dependent stage-two jobs.
            groups=[[j for j in jobs if j['row']['experiment']=='rffnet'],
                    [j for j in jobs if j['row']['experiment']!='rffnet']]
        else: groups=[jobs]
        for index,group in enumerate(groups):
            if group: run_jobs(o.root,group,o.gpus,o.resume,
                               o.root/f'queue_{o.phase}_{o.family}_{index}.csv')
    finally: guard.unlink(missing_ok=True)


if __name__=='__main__': main()
