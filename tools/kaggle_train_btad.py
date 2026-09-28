"""Explicit full training launch: GPU0=03, GPU1=01 then 02. No automatic evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--backbone',required=True)
    p.add_argument('--output-root',required=True)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=Path(args.output_root).resolve()
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'
    env['OMP_NUM_THREADS']='2'; env['MKL_NUM_THREADS']='2'
    env['CUDA_LAUNCH_BLOCKING']='0'
    def lane(categories,device):
        for category in categories:
            cmd=[sys.executable,'-m','external_baselines.inpformer_btad.run','train',
                 '--dataset-root',args.dataset_root,'--backbone',args.backbone,
                 '--category',category,'--device',device,'--workers','2',
                 '--output-dir',str(out/('train_'+category))]
            log=out/('train_'+category+'.log')
            print(f'START {category} on {device}; log={log}',flush=True)
            with log.open('w',encoding='utf-8') as f:
                result=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Category {category} failed; inspect {log}')
            print(f'COMPLETE {category}',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(lane,('03',),'cuda:0'),pool.submit(lane,('01','02'),'cuda:1')}
        errors=[]
        while pending:
            done,pending=wait(pending,timeout=60)
            for future in done:
                try: future.result()
                except Exception as exc:
                    errors.append(str(exc)); print(str(exc),flush=True)
            if pending:
                for log in sorted(out.glob('*.log')):
                    # Only read a short tail; training logs grow throughout the run.
                    with log.open('rb') as f:
                        f.seek(max(0,log.stat().st_size-4096))
                        lines=f.read().decode('utf-8',errors='replace').splitlines()
                    epochs=[s for s in lines if 'epoch [' in s or 'checkpoint saved:' in s]
                    if epochs: print(log.stem+': '+epochs[-1],flush=True)
        if errors: raise RuntimeError('\n'.join(errors))
    # Hard links provide a single inference root without duplicating large last.pt files.
    for category in ('01','02','03'):
        dest=out/'checkpoints'/category; dest.mkdir(parents=True)
        os.link(out/('train_'+category)/category/'last.pt',dest/'last.pt')
    (out/'complete.json').write_text(json.dumps(dict(stage='train',categories=['01','02','03'],
        checkpoint_root=str(out/'checkpoints'),epochs=200),indent=2),encoding='utf-8')
    print('All categories complete. Inference checkpoint root: '+str(out/'checkpoints'),flush=True)


if __name__=='__main__': main()
