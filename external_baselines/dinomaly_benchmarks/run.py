"""Independent phased entry: source/check/smoke/train/predict/evaluate/summarize/export."""
import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys
import traceback
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED
from .protocol import CONFIG, CATEGORIES, EXPECTED_COUNTS, OFFICIAL_ROOT, select
from . import official, data, checkpoint
from external_baselines.patchcore_official_eval.storage import json_write, csv_write, npz_write, read_json


def paths(args, category):
    return Path(args.output_dir).resolve()/args.dataset/category


def device(args):
    import torch
    value = torch.device(args.device)
    if value.type != 'cuda' or value.index is None or not torch.cuda.is_available():
        raise RuntimeError('Training, smoke and metric evaluation require an explicit available cuda:N')
    if value.index >= torch.cuda.device_count():
        raise RuntimeError('CUDA device unavailable')
    torch.cuda.set_device(value)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return value


def environment(out, value):
    import importlib.metadata
    from external_baselines.patchcore_official_eval.resources import environment as existing
    result = existing([str(value)])
    for name in ('ptflops','pandas','matplotlib','scikit-image','opencv-python-headless','tabulate','tqdm'):
        try:
            result['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['versions'][name] = 'not installed'
    json_write(out/'environment.json', result)


def identity(args, category, weights=None):
    return dict(config=CONFIG, dataset=args.dataset, category=category,
        dataset_root=str(Path(args.dataset_root).resolve()), backbone=weights,
        smoke=args.command == 'smoke')


def validate(saved, expected):
    if saved != expected:
        raise ValueError('Incompatible dataset/category/backbone/config/manifest; use original run settings')


def train(args, category, module, value, weight, smoke=False):
    import torch
    from external_baselines.destseg_mvtec_pretrained.resources import AllocatorResources
    root = data.adapter(args.dataset).find_root(args.dataset_root)
    # ImageFolder sorts filenames within the single normal class. Raw VisA CSV
    # row order is not that order; retain official training index ordering.
    items = sorted(data.image_records(args.dataset, root, category, 'train'), key=lambda r: Path(r.path).name)
    if smoke:
        items = items[:args.smoke_images]
    if len(items) < 16:
        raise ValueError('Official drop_last batch16 requires at least 16 normal training images')
    out = paths(args,category)/'train'
    out.mkdir(parents=True, exist_ok=True)
    ident = identity(args,category,weight)
    ident['training_manifest'] = [Path(r.path).relative_to(root).as_posix() for r in items]
    target = args.smoke_steps if smoke else 5000
    ident['target_updates'] = target
    ident['workers'] = args.workers
    if (out/'complete.json').exists():
        state = read_json(out/'complete.json')
        validate(state['identity'],ident)
        if state['status']!='complete' or state['completed_updates']!=target:
            raise ValueError('Invalid final training completion record')
        if args.skip_completed and (out/'final.pt').is_file():
            (out/'latest.pt').unlink(missing_ok=True)
            print(f'train {category}: already complete; skipped',flush=True)
            return
        raise FileExistsError(out/'complete.json')
    if (out/'identity.json').exists():
        validate(read_json(out/'identity.json'),ident)
        if not args.resume:
            raise FileExistsError('Incomplete training exists; explicitly use --resume')
    else:
        json_write(out/'identity.json',ident)
    attempt = out/f'attempt_{len(list(out.glob("attempt_*"))):04d}'
    attempt.mkdir(exist_ok=False)
    json_write(attempt/'config.json',dict(identity=ident,source=official.source(args.official_root),
        checkpoint_every=args.checkpoint_every, workers=args.workers, iterator='shuffle/drop_last; deterministic transforms; spawn workers',
        runtime='FP32; no AMP/TF32; no silent OOM fallback'))
    environment(attempt,value)
    meter = AllocatorResources(attempt,[value],category)
    module.setup_seed(1)  # Reset for EVERY category, including resume construction.
    with meter.measure('official_model_initialization'):
        model, trainable, optimizer, scheduler = official.build(module,args.official_root,args.backbone,value)
    transform,_ = module.get_data_transforms(448,392)
    loader = data.loader(items,transform,16,args.workers,training=True)
    state = None
    if (out/'latest.pt').exists():
        if not args.resume:
            raise FileExistsError('latest checkpoint exists; use --resume')
        state = torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        validate(state['identity'],ident)
        checkpoint.reload(model,state['model_without_encoder'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    completed = state['completed_updates'] if state else 0
    if not 0 <= completed <= target or scheduler.last_epoch != completed:
        raise ValueError('Optimizer update / scheduler counter mismatch')
    if state and completed != state['loader_cursor']['epoch']*len(loader)+state['loader_cursor']['batches']:
        raise ValueError('Optimizer update count disagrees with shuffled-loader progress')
    print(f'train {category}: {"resume" if state else "fresh initialization"}; completed_updates={completed}',flush=True)
    stream = checkpoint.Stream(loader,value,state['loader_cursor'] if state else None,state['rng'] if state else None)
    json_write(attempt/'resume.json',dict(resumed=state is not None,completed_updates=completed,
        epoch=stream.state['epoch'],consumed_batches=stream.state['batches'],
        recovery='model+optimizer+scheduler+RNG; epoch shuffle recreated and consumed batches replayed'))
    first_param = next(trainable.parameters())
    before = first_param.detach().clone() if smoke else None
    nonzero_lr = nonzero_gradient = False
    loss_path = attempt/'losses.csv'
    with meter.measure('training_and_checkpoint_save'), loss_path.open('w',encoding='utf-8',newline='') as log:
        writer = csv.DictWriter(log,fieldnames=['completed_updates','p','loss','lr_used','lr_next','gradient_norm_before_clip'])
        writer.writeheader()
        model.train()
        while completed < target:
            images = stream.next().to(value)
            en,de = model(images)
            p = min(.9*completed/1000,.9)
            loss = module.global_cosine_hm_percent(en,de,p=p,factor=.1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite official loss')
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable.parameters(),max_norm=.1,error_if_nonfinite=True)
            if any(p.grad is not None for p in model.encoder.parameters()):
                raise RuntimeError('Unexpected encoder gradient; official frozen forward contract changed')
            lr = float(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()
            completed += 1
            if scheduler.last_epoch != completed:
                raise ValueError('Scheduler no longer matches optimizer update count')
            nonzero_lr |= lr > 0
            nonzero_gradient |= float(norm) > 0
            writer.writerow(dict(completed_updates=completed,p=p,loss=float(loss),lr_used=lr,
                lr_next=float(optimizer.param_groups[0]['lr']),gradient_norm_before_clip=float(norm)))
            log.flush()
            if completed == 1 or completed%50 == 0 or completed == target:
                print(f'{category} step={completed}/{target} loss={float(loss):.6f} lr_used={lr:.8g}',flush=True)
            if completed%args.checkpoint_every == 0 or completed == target:
                checkpoint.save(out/'latest.pt',dict(identity=ident,model_without_encoder=checkpoint.compact(model),
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),completed_updates=completed,
                    rng=checkpoint.rng(value),loader_cursor=stream.state))
                json_write(out/'progress.json',dict(status='partial' if completed<target else 'finalizing',
                    completed_updates=completed,target_updates=target,latest='latest.pt'))
        if smoke and (not nonzero_lr or not nonzero_gradient or torch.equal(before,first_param.detach())):
            raise RuntimeError('Smoke must exercise nonzero LR/gradient and a real weight update')
        checkpoint.save(out/'final.pt',dict(identity=ident,completed_updates=completed,
            model_without_encoder=checkpoint.compact(model),frozen_encoder=weight))
        json_write(out/'complete.json',dict(status='complete',identity=ident,completed_updates=completed,
            checkpoint='final.pt',smoke=smoke,train_count=len(items)))
        # The completed fixed model stays; optimizer recovery state is only
        # necessary for unfinished classes. Avoid 27 duplicated optimizer banks.
        (out/'latest.pt').unlink()
    del stream,loader,model,trainable,optimizer,scheduler
    gc.collect()
    torch.cuda.empty_cache()
    print(f'train {category} complete; smoke={smoke}',flush=True)


def predict(args, category, module, value, weight, smoke=False):
    import torch
    import numpy as np
    from external_baselines.destseg_mvtec_pretrained.resources import AllocatorResources
    root = data.adapter(args.dataset).find_root(args.dataset_root)
    out = paths(args,category)/'predict'/args.prediction_name
    train_dir = paths(args,category)/'train'
    final = torch.load(train_dir/'final.pt',map_location='cpu',weights_only=False)
    expected = identity(args,category,weight)
    validate({k:final['identity'][k] for k in expected},expected)
    if final['completed_updates'] != (args.smoke_steps if smoke else 5000):
        raise ValueError('Only fixed final-step models may be predicted')
    ident = dict(run=expected,stage='predict',checkpoint='train/final.pt',name=args.prediction_name)
    if (out/'complete.json').exists():
        validate(read_json(out/'complete.json')['identity'],ident)
        if args.skip_completed:
            return
        raise FileExistsError(out)
    if (out/'identity.json').exists():
        validate(read_json(out/'identity.json'),ident)
        if not args.resume:
            raise FileExistsError('Incomplete prediction; use --resume or a new --prediction-name')
    out.mkdir(parents=True,exist_ok=True)
    json_write(out/'identity.json',ident)
    json_write(out/'config.json',dict(identity=ident,source=official.source(args.official_root),
        inference_batch=args.inference_batch,workers=args.workers,device=str(value),input='RGB image paths only'))
    environment(out,value)
    meter = AllocatorResources(out,[value],category)
    module.setup_seed(1)
    with meter.measure('strict_final_model_load'):
        model,trainable,optimizer,scheduler = official.build(module,args.official_root,args.backbone,value)
        checkpoint.reload(model,final['model_without_encoder'])
        del optimizer,scheduler,trainable
    items = data.image_records(args.dataset,root,category,'test')
    if smoke:
        items = items[:2]
    transform,_ = module.get_data_transforms(448,392)
    loader = data.loader(items,transform,args.inference_batch,args.workers)
    rows = []
    with meter.measure('inference_and_FP32_save'):
        for images in loader:
            images = images.to(value)
            maps,scores = official.score(model,images,module)
            if smoke:
                # Serialization check: disturb weights, strictly reload saved
                # compact state and compare real inference outputs in eval mode.
                with torch.no_grad():
                    next(model.bottleneck.parameters()).add_(.01)
                checkpoint.reload(model,final['model_without_encoder'])
                again,again_scores = official.score(model,images,module)
                torch.testing.assert_close(maps,again,rtol=0,atol=0)
                torch.testing.assert_close(scores,again_scores,rtol=0,atol=0)
            maps = maps.cpu().numpy()
            scores = scores.cpu().numpy()
            if maps.dtype != np.float32 or not np.isfinite(maps).all() or not np.isfinite(scores).all():
                raise FloatingPointError('Invalid FP32 predictions')
            for prediction,image_score in zip(maps,scores):
                record = items[len(rows)]
                row = data.metadata(record.path,root,category)
                relative = f'maps/{len(rows):06d}.npz'
                row.update(prediction_file=relative,image_score=float(image_score))
                npz_write(out/relative,anomaly_map=prediction,image_score=np.asarray(image_score,dtype=np.float32),
                    category=np.asarray(category),relative_path=np.asarray(row['relative_path']))
                # Save and reload actual arrays, not PNGs.
                with np.load(out/relative,allow_pickle=False) as cached:
                    np.testing.assert_array_equal(cached['anomaly_map'],prediction)
                    assert float(cached['image_score']) == row['image_score']
                rows.append(row)
            csv_write(out/'sample_scores.csv',rows)
            json_write(out/'samples.json',rows)
            print(f'predict {args.dataset}/{category}: {len(rows)}/{len(items)}',flush=True)
    json_write(out/'complete.json',dict(status='complete',identity=ident,image_count=len(rows),
        strict_reload=True,smoke=smoke,continuous_maps='FP32 NPZ; official cropped 256 grid'))
    del model,loader,final
    gc.collect()
    torch.cuda.empty_cache()


def check(args, module, categories):
    root,train_items,test_items = data.records(args.dataset,args.dataset_root,categories)
    out = Path(args.output_dir)/args.dataset/'data_check'
    out.mkdir(parents=True,exist_ok=False)
    counts,manifest = [],[]
    for category in categories:
        audits = []
        for split,items in [('train',train_items[category]),('test',test_items[category])]:
            for record in items:
                row = data.metadata(record.path,root,category)
                row.update(split=split,label=record.label,mask_path=record.mask_path)
                if split == 'test':
                    _,audit = data.gt(record,args.dataset,module)
                    row.update(audit)
                    if record.label:
                        audits.append(audit)
                manifest.append(row)
        counts.append(dict(category=category,train_normal=len(train_items[category]),test_count=len(test_items[category]),
            test_normal=sum(r.label==0 for r in test_items[category]),test_anomaly=sum(r.label==1 for r in test_items[category]),
            **{k:sum(a[k] for a in audits) for k in ('raw_empty','cropped_empty','evaluation_empty')}))
    actual = (sum(r['train_normal'] for r in counts),sum(r['test_count'] for r in counts))
    full = set(categories) == set(CATEGORIES[args.dataset])
    expected = EXPECTED_COUNTS[args.dataset] if full else None
    json_write(out/'config.json',dict(config=CONFIG,source=official.source(args.official_root)))
    environment(out,'cpu')
    csv_write(out/'counts.csv',counts)
    json_write(out/'manifest.json',manifest)
    json_write(out/'complete.json',dict(status='complete',actual_counts=actual,reference_counts=expected,
        difference=[a-b for a,b in zip(actual,expected)] if expected else None))
    print(json.dumps(dict(counts=counts,actual=actual,reference=expected),indent=2),flush=True)


def evaluate(args, category, module, value):
    import numpy as np
    from external_baselines.patchcore_official_eval.evaluation import calculate
    from external_baselines.destseg_mvtec_pretrained.resources import AllocatorResources
    pred = paths(args,category)/'predict'/args.prediction_name
    saved = read_json(pred/'complete.json')
    if saved['status'] != 'complete' or saved['smoke']:
        raise ValueError('Formal full-category cached predictions required')
    run = saved['identity']['run']
    if run['config'] != CONFIG or run['dataset'] != args.dataset or run['category'] != category:
        raise ValueError('Prediction protocol mismatch')
    if args.dataset_root is not None and run['dataset_root'] != str(Path(args.dataset_root).resolve()):
        raise ValueError('Use original dataset root for cache GT verification')
    out = paths(args,category)/'evaluate'/args.evaluation_name
    ident = dict(prediction_identity=saved['identity'],name=args.evaluation_name)
    if (out/'complete.json').exists():
        validate(read_json(out/'complete.json')['identity'],ident)
        if args.skip_completed:
            return
        raise FileExistsError(out)
    if (out/'identity.json').exists():
        validate(read_json(out/'identity.json'),ident)
        if not args.resume:
            raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    json_write(out/'identity.json',ident)
    json_write(out/'config.json',dict(identity=ident,source=official.source(args.official_root),
        device=str(value),evaluation=CONFIG['evaluation'],official_CPU_compute_pro_run=False))
    environment(out,value)
    meter = AllocatorResources(out,[value],category)
    with meter.measure('cached_maps_and_official_GT'):
        gt_manifest = pred/'ground_truth'/'manifest.json'
        if gt_manifest.exists():
            from types import SimpleNamespace
            lookup = {r['relative_path']:SimpleNamespace(label=r['label']) for r in read_json(gt_manifest)}
        else:
            if args.dataset_root is None:
                raise ValueError('First evaluation needs --dataset-root; later cached reevaluation does not')
            root,_,test = data.records(args.dataset,args.dataset_root,[category])
            lookup = {Path(r.path).relative_to(root).as_posix():r for r in test[category]}
        samples = read_json(pred/'samples.json')
        if len(samples)!=saved['image_count'] or len(samples)!=len(lookup) or {r['relative_path'] for r in samples}!=set(lookup):
            raise ValueError('Missing/duplicate test predictions')
        rows,maps,masks,audits = [],[],[],[]
        for index,row in enumerate(samples):
            with np.load(pred/row['prediction_file'],allow_pickle=False) as cached:
                image = cached['anomaly_map'].copy()
                if image.shape!=(256,256) or image.dtype!=np.float32 or not np.isfinite(image).all():
                    raise ValueError('Invalid cached continuous map')
                if (float(cached['image_score']) != row['image_score'] or
                    str(cached['relative_path']) != row['relative_path'] or str(cached['category']) != category):
                    raise ValueError('Prediction score/identity mismatch')
            record = lookup[row['relative_path']]
            row_metadata = {k:row[k] for k in ('category','original_hw','resize_hw','input_hw','crop_xyxy','evaluation_hw')}
            if (row_metadata['category']!=category or row_metadata['resize_hw']!=[448,448] or
                row_metadata['input_hw']!=[392,392] or row_metadata['crop_xyxy']!=[28,28,420,420] or
                row_metadata['evaluation_hw']!=[256,256]):
                raise ValueError('Cached prediction spatial contract mismatch')
            target = pred/'ground_truth'/f'{index:06d}.npz'
            if target.exists():
                with np.load(target,allow_pickle=False) as cached:
                    if str(cached['relative_path'])!=row['relative_path'] or int(cached['label'])!=record.label:
                        raise ValueError('GT cache identity mismatch')
                    mask = cached['mask'].copy()
                    audit = {k:bool(cached[k]) for k in ('raw_empty','cropped_empty','evaluation_empty')}
            else:
                if gt_manifest.exists():
                    raise FileNotFoundError('Cached GT is incomplete; restore the dataset for first evaluation')
                mask,audit = data.gt(record,args.dataset,module)
                npz_write(target,mask=mask,label=np.asarray(record.label),relative_path=np.asarray(row['relative_path']),**audit)
            if mask.dtype!=np.bool_ or mask.shape!=(256,256):
                raise ValueError('Invalid cropped GT cache')
            rows.append(dict(row,label=record.label))
            maps.append(image)
            masks.append(mask)
            audits.append(dict(category=category,relative_path=row['relative_path'],label=record.label,**audit))
        json_write(gt_manifest,[dict(relative_path=r['relative_path'],label=r['label']) for r in rows])
    with meter.measure('project_fast_CUDA_metrics'):
        result,fixed = calculate(rows,maps,masks,str(value),allow_cpu=False)
    json_write(out/'result.json',dict(category=category,metrics=result,fixed_fpr=fixed))
    csv_write(out/'category_metrics.csv',[dict(category=category,**result)])
    csv_write(out/'fixed_fpr.csv',[dict(category=category,**r) for r in fixed])
    csv_write(out/'mask_audit.csv',audits)
    json_write(out/'complete.json',dict(status='complete',identity=ident,image_count=len(rows),
        evaluation=CONFIG['evaluation'],mask_empty_counts={k:sum(r[k] for r in audits if r['label'])
        for k in ('raw_empty','cropped_empty','evaluation_empty')}))
    print(f'evaluate {args.dataset}/{category} complete',flush=True)


def summarize(args,categories):
    from external_baselines.patchcore_official_eval.evaluation import summarize as existing
    metrics,fixed = [],[]
    for category in categories:
        folder = paths(args,category)/'evaluate'/args.evaluation_name
        state = read_json(folder/'complete.json')
        run = state['identity']['prediction_identity']['run']
        if state['status']!='complete' or run['config']!=CONFIG or run['dataset']!=args.dataset or run['category']!=category or run['smoke']:
            raise ValueError('Incomplete/incompatible formal category')
        result = read_json(folder/'result.json')
        metrics.append(dict(category=category,**result['metrics']))
        fixed.extend(dict(category=category,**r) for r in result['fixed_fpr'])
    macro,fmacro = existing(metrics,fixed,categories,args.dataset)
    out = Path(args.output_dir)/args.dataset/'summaries'/args.evaluation_name
    if set(categories)!=set(CATEGORIES[args.dataset]):
        out = out/'selected'/'_'.join(categories)  # Never overwrite a previously complete full macro.
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('category_metrics',metrics),('fixed_fpr',fixed),('macro_metrics',macro),('fixed_fpr_macro',fmacro)]:
        csv_write(out/(filename+'.csv'),rows)
    json_write(out/'complete.json',dict(status='complete',categories=categories,scope=macro[0]['scope'],config=CONFIG))
    print(f"summary {args.dataset}: {macro[0]['scope']}; {len(categories)} classes",flush=True)


def export(args):
    root = Path(args.output_dir).resolve()/args.dataset
    destination = root.parent/f'Dinomaly_{args.dataset}_{args.export_kind}.zip'
    mode = ZIP_DEFLATED if args.export_kind=='report' else ZIP_STORED
    with ZipFile(destination,'w',mode) as z:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            keep = (path.suffix.lower() in ('.json','.jsonl','.csv','.log','.txt') if args.export_kind=='report'
                    else path.name=='final.pt' if args.export_kind=='weights'
                    else path.suffix=='.npz' if args.export_kind=='predictions'
                    else path.name=='latest.pt')
            if keep:
                z.write(path,path.relative_to(root.parent))
    print(destination,flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['source','check','smoke','train','predict','evaluate','summarize','export'])
    p.add_argument('--dataset',choices=['mvtec','visa'],required=True)
    p.add_argument('--dataset-root',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/dinomaly'))
    p.add_argument('--official-root',type=Path,default=OFFICIAL_ROOT)
    p.add_argument('--backbone',type=Path)
    p.add_argument('--categories',nargs='+',default=['all'])
    p.add_argument('--shard',choices=['0/2','1/2'])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--inference-batch',type=int,default=16)
    p.add_argument('--checkpoint-every',type=int,default=250)
    p.add_argument('--smoke-steps',type=int,default=2)
    p.add_argument('--smoke-images',type=int,default=32)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--skip-completed',action='store_true')
    p.add_argument('--prediction-name',default='v1')
    p.add_argument('--evaluation-name',default='v1')
    p.add_argument('--export-kind',choices=['report','weights','predictions','resume'],default='report')
    return p


def main():
    args = parser().parse_args()
    if args.workers<0 or min(args.checkpoint_every,args.inference_batch)<=0 or args.smoke_steps<2 or args.smoke_images<16:
        raise ValueError('Positive batch/checkpoint interval, workers>=0 and smoke>=2 steps/16 normals required')
    for name in (args.prediction_name,args.evaluation_name):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('Stage names must be simple directory names')
    categories = select(args.dataset,args.categories,args.shard)
    if args.command=='source':
        print(json.dumps(official.source(args.official_root),indent=2))
        return
    if args.command=='export':
        export(args)
        return
    if args.command=='summarize':
        summarize(args,categories)
        return
    if args.dataset_root is None and args.command!='evaluate':
        raise ValueError('--dataset-root is required')
    if args.command=='smoke':
        if len(categories)!=1:
            raise ValueError('Smoke runs exactly one selected category')
        if args.output_dir.exists():
            raise FileExistsError('Smoke requires a NEW separate output directory')
    module = official.symbols(args.official_root,args.dataset)
    if args.command=='check':
        check(args,module,categories)
        return
    value = device(args)
    weight = None
    if args.command in ('train','predict','smoke'):
        if args.backbone is None:
            raise ValueError('--backbone is required; automatic weight download is disabled')
        weight = official.weight_identity(args.backbone)
    for category in categories:
        try:
            if args.command in ('train','smoke'):
                train(args,category,module,value,weight,smoke=args.command=='smoke')
            if args.command in ('predict','smoke'):
                predict(args,category,module,value,weight,smoke=args.command=='smoke')
            if args.command=='evaluate':
                evaluate(args,category,module,value)
        except BaseException as exc:
            json_write(paths(args,category)/f'{args.command}_failure.json',dict(status='failed',type=type(exc).__name__,
                message=str(exc),traceback=traceback.format_exc(),configuration_changed=False))
            raise


if __name__=='__main__':
    main()
