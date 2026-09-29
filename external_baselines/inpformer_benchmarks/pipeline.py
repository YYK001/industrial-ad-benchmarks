"""Kaggle two-GPU execution with per-category resume, evaluation and lightweight ZIPs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import zipfile

from .data import Adapter, discover
from external_baselines.inpformer_btad.run import write_json, write_csv

REPO=Path(__file__).resolve().parents[2]
METRICS=('image_AUROC','image_AUPR','pixel_AUROC','pixel_AUPR','AUPRO_at_0p3')


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def complete(path, stage, dataset, category):
    marker=path/'complete.json'
    if not marker.exists(): return False
    state=read_json(marker)
    if state.get('protocol')!=Adapter(dataset).PROTOCOL or state.get('stage')!=stage or state.get('categories')!=[category]:
        raise ValueError(f'incompatible completion marker: {marker}')
    if stage=='train' and not (path/category/'last.pt').is_file(): raise FileNotFoundError(path/category/'last.pt')
    return True


def completed_attempt(folder, stage, dataset, category):
    found=[p for p in sorted(folder.glob(stage+'_[0-9]*')) if p.is_dir() and complete(p,stage,dataset,category)]
    if len(found)>1: raise RuntimeError(f'multiple completed attempts, choose explicitly: {found}')
    return found[0] if found else None


def fresh_attempt(folder, stage):
    folder.mkdir(parents=True,exist_ok=True)
    i=0
    while (folder/f'{stage}_{i:03d}').exists(): i+=1
    return folder/f'{stage}_{i:03d}'


def latest_resume(folder, category):
    candidates=[]
    for p in folder.glob(f'train_[0-9]*/{category}/resume_latest.pt'):
        progress=p.parent/'checkpoint_progress.json'
        try:
            epoch=int(read_json(progress)['epoch'])
        except (OSError, ValueError, KeyError):
            import torch
            epoch=int(torch.load(p,map_location='cpu',weights_only=False)['epoch'])
        candidates.append((epoch,p.stat().st_mtime_ns,p))
    return max(candidates)[2] if candidates else None


def prune_resume(folder, category, keep):
    # Only rolling files created under this category's attempts; no old experiment directories.
    root=folder.resolve()
    for p in folder.glob(f'train_[0-9]*/{category}/resume_latest.pt'):
        p.resolve().relative_to(root)
        if keep is None or p!=keep: p.unlink()


@contextmanager
def execution_lock(out):
    if os.name!='posix': raise RuntimeError('formal pipeline launcher requires Linux/Kaggle; CPU unit tests are cross-platform')
    import fcntl
    with (out/'pipeline.lock').open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('this output has a running launcher/child; do not start duplicate training')
        # Child subprocesses inherit this descriptor: even orphaned training retains the lock.
        yield f.fileno()


def locate(args):
    datasets=('mvtec','visa') if args.dataset=='both' else (args.dataset,)
    detected=None; roots={}
    for name in datasets:
        explicit=getattr(args,name+'_root')
        if explicit: roots[name]=str(Adapter(name).find_root(explicit)); continue
        if detected is None: detected=discover(args.input_root)
        if len(detected[name])!=1:
            raise ValueError(f'{name}: need --{name}-root; candidates={detected[name]}')
        roots[name]=str(Adapter(name).find_root(detected[name][0]))
    return roots


def plan(args):
    roots=locate(args); tasks=[]
    if args.categories and len(roots)!=1: raise ValueError('--categories requires one dataset')
    for dataset,root in roots.items():
        adapter=Adapter(dataset)
        categories=tuple(args.categories or adapter.CATEGORIES)
        if len(set(categories))!=len(categories) or not set(categories)<=set(adapter.CATEGORIES): raise ValueError('unknown/duplicate categories')
        for c in categories:
            train=adapter.image_records(root,c,'train'); test=adapter.image_records(root,c,'test')
            if len(train)<32: raise ValueError(f'{dataset}/{c} has fewer than 32 normal images')
            tasks.append(dict(dataset=dataset,category=c,root=root,layout=adapter.layout(root),train_count=len(train),
                              test_count=len(test),total_steps=200*(len(train)//16)))
    return roots,tasks


def aggregate(dataset, categories, completed, output):
    metrics=[]; fixed=[]; counts=[]; audits=[]; resources=[]
    for c in categories:
        paths=completed[c]
        metrics.extend(read_csv(paths['evaluate']/'category_metrics.csv'))
        fixed.extend(read_csv(paths['evaluate']/'fixed_fpr.csv'))
        counts.extend(read_csv(paths['evaluate']/'counts.csv'))
        if (paths['evaluate']/'mask_audit.csv').exists(): audits.extend(read_csv(paths['evaluate']/'mask_audit.csv'))
        for stage,path in paths.items():
            if (path/'resources.csv').exists():
                resources.extend(dict(pipeline_stage=stage,**row) for row in read_csv(path/'resources.csv'))
    if sorted(r['category'] for r in metrics)!=sorted(categories): raise ValueError('missing/duplicate category metric rows')
    macro=dict(scope='full_dataset_equal_macro' if set(categories)==set(Adapter(dataset).CATEGORIES) else 'selected_categories_equal_macro',
               category_count=len(categories),**{k:sum(float(r[k]) for r in metrics)/len(metrics) for k in METRICS})
    fm=[]
    for cap in (.01,.05):
        subset=[r for r in fixed if float(r['fpr_cap'])==cap]
        if len(subset)!=len(categories): raise ValueError('fixed FPR category count')
        row=dict(fpr_cap=cap,category_count=len(subset))
        for key in ('defect_pixel_recall','region_mean_coverage','small_region_mean_coverage'):
            values=[float(r[key]) for r in subset if r[key]!='N/A']
            row[key]=sum(values)/len(values) if values else 'N/A'; row[key+'_category_count']=len(values)
        fm.append(row)
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in dict(category_metrics=metrics,macro_metrics=[macro],fixed_fpr=fixed,fixed_fpr_macro=fm,
                          counts=counts,mask_audit=audits,resources=resources).items(): write_csv(output/(name+'.csv'),rows)
    lines=['# '+Adapter(dataset).TITLE,'','200轮最后模型；类别等权macro；AP为average precision。',
        '官方448→392中心裁剪、256×256评价。不得与完整视野结果直接计算差值。',
        'AUPRO使用既有CUDA实现、200阈值、FPR上限0.3；固定FPR为测试集事后诊断，不是部署误报率。',
        '小区域≤256×256面积的0.1%，四连通；N/A不参与该项macro。以下五指标按百分比显示。','',
        '|类别|Image AUROC|Image AP|Pixel AUROC|Pixel AP|AUPRO@0.3|',
        '|---|---:|---:|---:|---:|---:|']
    lines.extend('|'+r['category']+'|'+'|'.join(f'{float(r[k])*100:.4f}' for k in METRICS)+'|' for r in metrics)
    lines+=['|macro|'+'|'.join(f'{macro[k]*100:.4f}' for k in METRICS)+'|','',
        '耗时汇总仅统计各类成功训练尝试；若中断过，全部尝试的资源记录在ZIP中另行保留，不能把成功尝试耗时当作从头训练总耗时。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(output/'complete.json',dict(dataset=dataset,categories=list(categories),macro=macro))


def package_results(dataset_root, summary, archive):
    # Whitelist small objective records. No .pt/.npy, images, secrets, or training log bulk.
    candidates=set(p for p in dataset_root.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.md'))
    # Retain one actual source snapshot for reproducibility, rather than one per stage/category.
    snapshots=sorted(dataset_root.glob('*/evaluate_[0-9]*/source_snapshot'))
    if snapshots: candidates.update(p for p in snapshots[0].rglob('*.py') if p.is_file())
    temporary=archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(candidates): z.write(p,p.relative_to(dataset_root).as_posix())
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC failure')
    temporary.replace(archive)


def run(args):
    roots,tasks=plan(args)
    if args.command=='inspect':
        print(json.dumps(dict(roots=roots,tasks=tasks,total_steps=sum(t['total_steps'] for t in tasks)),ensure_ascii=False,indent=2)); return
    if not args.backbone or not Path(args.backbone).is_file(): raise FileNotFoundError('--backbone is required for run')
    if not args.output_root: raise ValueError('--output-root required for run')
    out=Path(args.output_root).resolve(); out.mkdir(parents=True,exist_ok=True)
    identity=dict(roots=roots,tasks=tasks,backbone=str(Path(args.backbone).resolve()),workers=2)
    if (out/'plan.json').exists() and read_json(out/'plan.json')!=identity:
        raise ValueError('output belongs to a different plan/paths; use its original settings or another output root')
    with execution_lock(out) as lockfd:
        write_json(out/'plan.json',identity)
        env=os.environ.copy(); env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_LAUNCH_BLOCKING='0')
        env['PYTHONPATH']=str(REPO)+os.pathsep+env.get('PYTHONPATH','')
        live=set(); mutex=threading.Lock(); cancel=threading.Event()
        def stage(command, task, device, destination, extras=()):
            if cancel.is_set(): raise RuntimeError('launch cancelled')
            args2=[sys.executable,'-u','-m','external_baselines.inpformer_benchmarks.run','--dataset',task['dataset'],command,
                   '--dataset-root',task['root'],'--category',task['category'],'--device',device,'--workers','2',
                   '--output-dir',str(destination),*extras]
            destination.parent.mkdir(parents=True,exist_ok=True)
            log=destination.with_suffix('.log')
            print(f"START {task['dataset']}/{task['category']} {command} on {device}; {log}",flush=True)
            with log.open('w',encoding='utf-8') as f:
                with mutex:
                    if cancel.is_set(): raise RuntimeError('launch cancelled')
                    proc=subprocess.Popen(args2,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,
                                          start_new_session=True,pass_fds=(lockfd,)); live.add(proc)
                try: code=proc.wait()
                finally:
                    with mutex: live.discard(proc)
            if code: raise RuntimeError(f'{command} failed ({code}); inspect {log}')
        work=queue.Queue()
        for task in sorted(tasks,key=lambda t:t['total_steps'],reverse=True): work.put(task)
        finished={k:{} for k in roots}
        def lane(device):
            while not cancel.is_set():
                try: task=work.get_nowait()
                except queue.Empty: return
                dataset=task['dataset']; c=task['category']; folder=out/dataset/c
                paths={}
                train=completed_attempt(folder,'train',dataset,c)
                if train is None:
                    resume=latest_resume(folder,c); prune_resume(folder,c,resume)
                    train=fresh_attempt(folder,'train')
                    extras=['--backbone',args.backbone]+(['--resume',str(resume)] if resume else [])
                    print(f'{dataset}/{c}: '+(f'resume {resume}' if resume else 'fresh model; no recovery point'),flush=True)
                    try: stage('train',task,device,train,extras)
                    finally: prune_resume(folder,c,latest_resume(folder,c))
                prune_resume(folder,c,None); paths['train']=train
                prediction=completed_attempt(folder,'predict',dataset,c)
                if prediction is None:
                    prediction=fresh_attempt(folder,'predict')
                    stage('predict',task,device,prediction,['--backbone',args.backbone,'--checkpoint-root',str(train)])
                paths['predict']=prediction
                evaluation=completed_attempt(folder,'evaluate',dataset,c)
                if evaluation is None:
                    evaluation=fresh_attempt(folder,'evaluate')
                    stage('evaluate',task,device,evaluation,['--prediction-root',str(prediction)])
                paths['evaluate']=evaluation
                with mutex: finished[dataset][c]=paths
                print(f'COMPLETE {dataset}/{c}',flush=True)
        pool=ThreadPoolExecutor(max_workers=2)
        try:
            pending={pool.submit(lane,'cuda:0'),pool.submit(lane,'cuda:1')}
            while pending:
                done,pending=wait(pending,timeout=60)
                for future in done: future.result()
                if pending:
                    for log in out.glob('*/*/train_[0-9]*.log'):
                        with log.open('rb') as f:
                            f.seek(max(0,log.stat().st_size-3000)); lines=f.read().decode('utf-8',errors='replace').splitlines()
                        progress=[s for s in lines if 'epoch [' in s or 'checkpoint saved:' in s]
                        if progress: print(str(log.relative_to(out))+': '+progress[-1],flush=True)
        except BaseException:
            cancel.set()
            with mutex:
                for proc in list(live):
                    try: os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError: pass
            raise
        finally: pool.shutdown(wait=True)
        for dataset in roots:
            categories=[t['category'] for t in tasks if t['dataset']==dataset]
            summary=out/dataset/'summary'; aggregate(dataset,categories,finished[dataset],summary)
            archive=out/f'INPFormer_{dataset}_objective_results.zip'
            package_results(out/dataset,summary,archive)
            print('DOWNLOAD: '+str(archive),flush=True)
        write_json(out/'complete.json',dict(stage='train_predict_evaluate_package',datasets=list(roots)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('inspect','run'))
    p.add_argument('--dataset',choices=('mvtec','visa','both'),default='both')
    p.add_argument('--input-root',default='/kaggle/input')
    p.add_argument('--mvtec-root'); p.add_argument('--visa-root')
    p.add_argument('--categories',nargs='+')
    p.add_argument('--backbone'); p.add_argument('--output-root')
    run(p.parse_args())


if __name__=='__main__': main()
