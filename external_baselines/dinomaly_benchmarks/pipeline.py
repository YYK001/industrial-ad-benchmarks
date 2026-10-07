"""Two independent T4 processes by category; resume existing outputs explicitly."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from .protocol import OFFICIAL_ROOT, select
from .run import export, summarize
from external_baselines.patchcore_official_eval.storage import json_write
from external_baselines.destseg_visa.processes import run_process


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['mvtec','visa'],required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--official-root',type=Path,default=OFFICIAL_ROOT)
    p.add_argument('--backbone',type=Path,required=True)
    p.add_argument('--categories',nargs='+',default=['all'])
    p.add_argument('--devices',nargs=2,default=['cuda:0','cuda:1'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--inference-batch',type=int,default=16)
    p.add_argument('--checkpoint-every',type=int,default=250)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--idle-timeout',type=int,default=900)
    args = p.parse_args()
    if len(set(args.devices))!=2 or args.workers<0 or args.idle_timeout<=0:
        p.error('Two different CUDA devices, workers>=0 and positive timeout required')
    categories = select(args.dataset,args.categories)
    logs = args.output_dir/args.dataset/'launcher'
    logs.mkdir(parents=True,exist_ok=True)
    cancel = threading.Event()
    completed = []
    lock = threading.Lock()
    json_write(logs/'config.json',dict(dataset=args.dataset,categories=categories,
        devices=args.devices,groups=[categories[::2],categories[1::2]],resume=args.resume))
    def worker(index):
        for category in categories[index::2]:
            if cancel.is_set():
                return
            for stage in ('train','predict','evaluate'):
                command = [sys.executable,'-u','-m','external_baselines.dinomaly_benchmarks',stage,
                    '--dataset',args.dataset,'--dataset-root',str(args.dataset_root),
                    '--output-dir',str(args.output_dir),'--official-root',str(args.official_root),
                    '--backbone',str(args.backbone),'--categories',category,'--device',args.devices[index],
                    '--workers',str(args.workers),'--inference-batch',str(args.inference_batch),
                    '--checkpoint-every',str(args.checkpoint_every),'--skip-completed']
                if args.resume:
                    command.append('--resume')
                tag = f'{stage}_{category}_gpu{index}'
                run_process(command,logs/(tag+'.log'),tag,cancel,
                    idle_timeout=3600 if stage=='evaluate' else args.idle_timeout)
            with lock:
                completed.append(category)
                json_write(logs/'progress.json',dict(completed=completed,pending=[c for c in categories if c not in completed]))
    status = 'failed'
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(worker,i) for i in range(2)]
            try:
                for job in jobs:
                    job.result()
            except BaseException:
                cancel.set()  # Signal before pool.__exit__ waits for workers.
                raise
        args.evaluation_name = 'v1'
        summarize(args,categories)  # Missing categories fail; never write a false full macro.
        status = 'complete'
    finally:
        cancel.set()
        json_write(logs/'pipeline_status.json',
            dict(status=status,categories=categories,completed=completed,pending=[c for c in categories if c not in completed]))
        args.export_kind='report'
        export(args)  # Text-only partial/complete records, no weights/maps/environment tree.


if __name__=='__main__':
    main()
