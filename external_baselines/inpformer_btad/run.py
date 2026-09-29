"""Independent check / resource smoke / train / predict / evaluate commands."""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import random
import subprocess
import time

import numpy as np

from .data import CATEGORIES, SPATIAL, find_root, image_records, geometry, load_gt, records
from .model import external, symbols, build, predict_batches

TITLE = 'INP-Former官方单类full-shot配置的BTAD外部复评'
PROTOCOL = 'inpformer_btad_single_class_fullshot_seed1_crop_v1'
DATASET_NAME = 'BTAD'
COMPACT_FINAL = False
FULL_MACRO_SCOPE = 'three_category_equal_macro'
CONFIG = dict(external.OFFICIAL_CONFIG, seed=1, precision='FP32', shuffle=True, drop_last=True,
              optimizer='StableAdamW', lr=1e-3, betas=[.9, .999], weight_decay=1e-4,
              amsgrad=True, eps=1e-10, scheduler='WarmCosineScheduler', final_lr=1e-4,
              warmup_iters=100, loss='global_cosine_hm_adaptive(y=3) + 0.2*g_loss',
              clip_grad_max_norm=.1, checkpoint='last_epoch_only', spatial=SPATIAL)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def selected(args):
    return CATEGORIES if args.category == 'all' else (args.category,)


def source_state(args):
    root = Path(args.official_root).resolve()
    head = external.read_git_head_without_git(root)
    if head != external.EXPECTED_OFFICIAL_COMMIT:
        raise ValueError('unreviewed official source version: '+head)
    project=Path(__file__).resolve().parents[2]
    revision=None; dirty=None
    try:
        revision=external.read_git_head_without_git(project)
        status=subprocess.run(['git','status','--porcelain'],cwd=project,capture_output=True,text=True,check=True)
        dirty=bool(status.stdout.strip())
    except (OSError,RuntimeError,subprocess.SubprocessError):
        pass  # The actual source snapshot remains available without Git metadata.
    return dict(title=TITLE, protocol=PROTOCOL, config=CONFIG,
                adapter_revision=revision, adapter_worktree_dirty=dirty, dataset=DATASET_NAME,
                final_checkpoint_format='model_without_encoder' if COMPACT_FINAL else 'full_model',
                official_commit=head, official_root=str(root),
                backbone_sha256=external.EXPECTED_BACKBONE_SHA256,
                external_reuse='external_baselines/inpformer_external/run.py',
                python=platform.python_version(), packages=external.package_versions(),
                arguments=vars(args), cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING', 'unset'),
                progress_checkpoints=dict(every_epochs=20, rolling_resume=True, keep_history=False,
                                          remove_resume_after_final=True, evaluation_epoch=200),
                runtime=dict(num_workers=args.workers, pin_memory=True,
                             persistent_workers=args.workers>0, prefetch_factor=2 if args.workers else None,
                             tf32=False, autocast=False))


def save_config(out, args):
    """Small source snapshot, no data/weights/features or hash closure."""
    write_json(out/'config.json', source_state(args))
    project = Path(__file__).resolve().parents[2]
    official = Path(args.official_root).resolve()
    sources = [(p, Path('official')/p.relative_to(official)) for p in official.rglob('*.py')
               if '.git' not in p.parts]
    for folder in (Path(__file__).parent, project/'external_baselines/inpformer_external',
                   project/'external_baselines/inpformer_benchmarks'):
        sources.extend((p, Path(folder.name)/p.name) for p in folder.glob('*.py'))
    for relative in ('DINOv3/MADEqual/btad_validation/dataset.py',
                     'DINOv3/MADEqual/visa_task_decoupled/dataset.py',
                     'DINOv3/relation_reliability/datasets.py',
                     'DINOv3/MADEqual/mvtec_broad6_compose2/metrics.py',
                     'DINOv3/MADEqual/hard_sample_discrimination/scoring.py',
                     'DINOv3/nsrm/m0_scoring.py'):
        if (project/relative).is_file(): sources.append((project/relative, Path(relative)))
    for source, relative in sources:
        target = out/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


class Meter:
    def __init__(self, out, device):
        self.out, self.device, self.rows = out, device, []

    @contextlib.contextmanager
    def measure(self, stage, category):
        import torch
        cuda = self.device.type == 'cuda'
        if cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize(self.device)
            self.rows.append(dict(stage=stage, category=category, seconds=time.perf_counter()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device) if cuda else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device) if cuda else 0))
            write_csv(self.out/'resources.csv', self.rows)


def cuda(args):
    import torch
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('resource smoke and formal stages require real CUDA; use check/tests on CPU')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def check(args):
    root, train, test, _, _ = records(args.dataset_root, selected(args), inspect_masks=True)
    out = fresh(args.output_dir)
    save_config(out, args)
    rows, listing = [], []
    for c in selected(args):
        masks = [load_gt(r)[1] for r in test[c] if r.label == 1]
        rows.append(dict(category=c, train_normal=len(train[c]), test_count=len(test[c]),
                         test_normal=sum(r.label == 0 for r in test[c]),
                         test_anomaly=sum(r.label == 1 for r in test[c]),
                         **{k:sum(m[k] for m in masks) for k in ('raw_empty','cropped_empty','evaluation_empty')}))
        for split, items in (('train', train[c]), ('test', test[c])):
            pure = image_records(root, c, split)
            if [r.path for r in items] != [r.path for r in pure]:
                raise ValueError('BTAD test universe/order mismatch')
            for r in items:
                listing.append(dict(category=c, split=split, relative_path=Path(r.path).relative_to(root).as_posix(),
                    label=r.label, mask=Path(r.mask_path).relative_to(root).as_posix() if r.mask_path else '',
                    role='fullshot_normal_training' if split=='train' else 'evaluation', geometry=geometry(r.path)))
    write_csv(out/'counts.csv', rows)
    write_json(out/'input_manifest.json', listing)
    write_json(out/'complete.json', dict(stage='check', counts=rows, status='complete', protocol=PROTOCOL))
    print(json.dumps(rows, indent=2))


def save_checkpoint(path, model, category, count, smoke=False):
    import torch
    weights = ({'model_without_encoder': {k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')}}
               if COMPACT_FINAL else {'model':model.state_dict()})
    torch.save(dict(protocol=PROTOCOL, category=category, config=CONFIG,
                    epoch=1 if smoke else 200, smoke=smoke, train_count=count,
                    **weights), path)


def restore_model(model, state):
    if 'model' in state:
        model.load_state_dict(state['model'], strict=True)
        return
    current=model.state_dict()
    expected={k for k in current if not k.startswith('encoder.')}
    if set(state['model_without_encoder'])!=expected:
        raise ValueError('compact checkpoint keys mismatch')
    current.update(state['model_without_encoder'])
    model.load_state_dict(current,strict=True)


def load_checkpoint(path, model, category):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('protocol') != PROTOCOL or state.get('category') != category or
        state.get('config') != CONFIG or state.get('smoke') or state.get('epoch') != 200):
        raise ValueError('requires this protocol category final epoch 200 checkpoint; smoke/RobustAD forbidden')
    restore_model(model, state)
    return state['train_count']


def train(args):
    import torch
    device = cuda(args); root = find_root(args.dataset_root); out = fresh(args.output_dir)
    save_config(out, args); meter = Meter(out, device)
    value = symbols(args.official_root, args.backbone)
    transform, _ = value['get_data_transforms'](448, 392)
    for c in selected(args):
        items = image_records(root, c, 'train')
        value['setup_seed'](1)
        with meter.measure('initialization', c):
            model, trainable = build(value, args.official_root, device)
        dest = out/c; dest.mkdir()
        identifiers = [Path(r.path).relative_to(root).as_posix() for r in items]
        write_json(dest/'training_inputs.json', identifiers)
        resume = None
        if args.resume:
            if args.category == 'all': raise ValueError('--resume requires one explicit category')
            resume = torch.load(args.resume, map_location='cpu', weights_only=False)
            if (resume.get('protocol') != PROTOCOL or resume.get('category') != c or
                resume.get('config') != CONFIG or resume.get('training_inputs') != identifiers or
                resume.get('workers') != args.workers or resume.get('kind') != 'epoch_resume' or
                not 0 < resume.get('epoch', 0) <= 200):
                raise ValueError('incompatible resume checkpoint/category/training inputs/workers')
            weights = model.state_dict()
            expected = {k for k in weights if not k.startswith('encoder.')}
            if set(resume['model_without_encoder']) != expected:
                raise ValueError('resume trainable model keys mismatch')
            weights.update(resume['model_without_encoder'])
            model.load_state_dict(weights, strict=True)
            del weights
        def checkpoint_epoch(epoch, history, optimizer, scheduler):
            write_csv(dest/'epoch_losses.csv', history)
            if epoch % 20: return
            # Frozen encoder is reloaded from the verified pretrained file on resume.
            weights = {k:v for k,v in model.state_dict().items() if not k.startswith('encoder.')}
            payload = dict(protocol=PROTOCOL, category=c, config=CONFIG, epoch=epoch,
                           model_without_encoder=weights, train_count=len(items), smoke=False)
            temporary = dest/'checkpoint.tmp'
            payload.update(kind='epoch_resume', optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                           history=list(history), training_inputs=identifiers, workers=args.workers,
                           python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                           torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device))
            torch.save(payload, temporary)
            temporary.replace(dest/'resume_latest.pt')
            write_json(dest/'checkpoint_progress.json', dict(epoch=epoch, filename='resume_latest.pt'))
            print(f'checkpoint saved: {c} epoch {epoch}/200', flush=True)
        with meter.measure('training', c):
            history = external.train_model(model, trainable, items, transform, value, device, 200, 16, args.workers,
                                           resume_state=resume, epoch_callback=checkpoint_epoch)
        write_csv(dest/'epoch_losses.csv', history)
        with meter.measure('checkpoint_storage', c):
            temporary = dest/'last.tmp'
            save_checkpoint(temporary, model, c, len(items))
            temporary.replace(dest/'last.pt')
            # Keep the recovery point until the final checkpoint is fully written.
            (dest/'resume_latest.pt').unlink(missing_ok=True)
        write_json(dest/'complete.json', dict(protocol=PROTOCOL, category=c, epoch=200, train_count=len(items),
                   steps_per_epoch=len(items)//16, scheduler_total_iters=200*(len(items)//16)))
        del model, trainable; gc.collect(); torch.cuda.empty_cache()
    write_json(out/'complete.json', dict(protocol=PROTOCOL, stage='train', categories=selected(args)))


def predict(args):
    import torch
    device = cuda(args); root = find_root(args.dataset_root); out = fresh(args.output_dir)
    save_config(out, args); meter = Meter(out, device)
    value = symbols(args.official_root, args.backbone)
    transform, _ = value['get_data_transforms'](448, 392)
    for c in selected(args):
        items = image_records(root, c, 'test')
        value['setup_seed'](1)
        with meter.measure('initialization_and_checkpoint_load', c):
            model, trainable = build(value, args.official_root, device)
            count = load_checkpoint(Path(args.checkpoint_root)/c/'last.pt', model, c)
        dest = out/c; dest.mkdir(); rows = []
        with meter.measure('inference_including_prediction_storage', c):
            for offset, maps, scores in predict_batches(model, items, transform, value, device, args.workers):
                for j, (pixel, score) in enumerate(zip(maps, scores)):
                    i = offset+j; relative = Path(items[i].path).relative_to(root).as_posix()
                    np.save(dest/f'{i:04d}.npy', pixel, allow_pickle=False)
                    rows.append(dict(index=i, relative_path=relative, image_score=float(score),
                                     prediction=f'{i:04d}.npy', geometry=geometry(items[i].path)))
        write_json(dest/'predictions.json', rows)
        write_csv(dest/'image_scores.csv', [{k:v for k,v in r.items() if k!='geometry'} for r in rows])
        write_json(dest/'complete.json', dict(protocol=PROTOCOL, stage='predict', category=c,
                   train_count=count, test_count=len(items), checkpoint_epoch=200, spatial=SPATIAL))
        del model, trainable; gc.collect(); torch.cuda.empty_cache()
    write_json(out/'complete.json', dict(protocol=PROTOCOL, stage='predict', categories=selected(args)))


def evaluate(args):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import evaluate_fast, METRIC_NAMES
    from DINOv3.MADEqual.hard_sample_discrimination.scoring import fixed_fpr_diagnostics
    device = cuda(args); root = find_root(args.dataset_root); out = fresh(args.output_dir)
    meter = Meter(out, device); metrics, operating, audits, counts = [], [], [], []
    save_config(out, args)
    for c in selected(args):
        dest = Path(args.prediction_root)/c
        state = json.loads((dest/'complete.json').read_text(encoding='utf-8'))
        if state.get('protocol') != PROTOCOL or state.get('checkpoint_epoch') != 200 or state.get('spatial') != SPATIAL:
            raise ValueError('incompatible/incomplete prediction set')
        with meter.measure('evaluation_inputs_and_gt_read', c):
            _, _, test, _, _ = records(root, (c,), inspect_masks=False)
            rows = json.loads((dest/'predictions.json').read_text(encoding='utf-8'))
            items = test[c]
            if [r['relative_path'] for r in rows] != [Path(r.path).relative_to(root).as_posix() for r in items]:
                raise ValueError('prediction/test universe mismatch')
            maps, masks, scores = [], [], []
            for row, record in zip(rows, items):
                if row['geometry'] != geometry(record.path):
                    raise ValueError('prediction geometry mismatch')
                pixel = np.load(dest/row['prediction'], allow_pickle=False)
                if pixel.shape != (256,256) or pixel.dtype != np.float32 or not np.isfinite(pixel).all():
                    raise ValueError('invalid prediction map')
                mask, audit = load_gt(record)
                masks.append(mask); maps.append(pixel); scores.append(row['image_score'])
                if record.label == 1:
                    audits.append(dict(category=c, relative_path=row['relative_path'], **audit))
            labels = [r.label for r in items]
        with meter.measure('metric_evaluation', c):
            result = evaluate_fast(labels, masks, maps, device=device, allow_cpu_fallback=False)
            # evaluate_fast's default image maximum is replaced by the persisted official top-1% score.
            result.update(image_AUROC=float(roc_auc_score(labels, scores)),
                          image_AUPR=float(average_precision_score(labels, scores)))
            metrics.append(dict(category=c, **result))
            operating.extend(dict(category=c, **r) for r in fixed_fpr_diagnostics(maps, masks))
        counts.append(dict(category=c, train_normal=state['train_count'], test_count=len(items),
            **{k:sum(r[k] for r in audits if r['category']==c) for k in ('raw_empty','cropped_empty','evaluation_empty')}))
        del maps, masks; gc.collect()
    macro = [dict(scope=FULL_MACRO_SCOPE if tuple(selected(args))==tuple(CATEGORIES) else 'selected_category_equal_macro',
                  category_count=len(metrics), **{k:float(np.mean([r[k] for r in metrics])) for k in METRIC_NAMES})]
    fixed_macro = []
    for cap in (.01, .05):
        subset = [r for r in operating if r['fpr_cap']==cap]
        row = dict(fpr_cap=cap, category_count=len(subset))
        for key in ('defect_pixel_recall', 'region_mean_coverage', 'small_region_mean_coverage'):
            vals = [r[key] for r in subset if r[key] != 'N/A']
            row[key] = float(np.mean(vals)) if vals else 'N/A'
            row[key+'_category_count'] = len(vals)
        fixed_macro.append(row)
    for name, rows in dict(category_metrics=metrics, macro_metrics=macro, fixed_fpr=operating,
                           fixed_fpr_macro=fixed_macro, mask_audit=audits, counts=counts).items():
        write_csv(out/(name+'.csv'), rows)
    (out/'report.md').write_text('# '+TITLE+'\n\n官方中心裁剪视野，256×256 评价。不能与完整视野指标直接计算差值。\n'
        '各类独立正常 full-shot，200 轮最后 checkpoint；本报告使用当前环境和既有CUDA指标，不主张原论文指标逐位复现。\n'
        'AUPR 字段均为 average precision (AP)。AUPRO 复用快速 CUDA，200 阈值，FPR≤0.3。\n'
        '固定 FPR 1%/5% 为测试集事后诊断，不是部署误报率。四连通，小区域≤本次 256×256 面积的 0.1%。\n'
        '空 mask 异常保留图像标签；小区域 N/A 不参与该项 macro。\n\n'
        '五指标与类别等权 macro 见 category_metrics.csv、macro_metrics.csv；固定 FPR 见 fixed_fpr*.csv。\n', encoding='utf-8')
    write_json(out/'complete.json', dict(protocol=PROTOCOL, stage='evaluate', categories=selected(args), macro=macro))


def smoke(args):
    """Two normal-only batches, including a nonzero-LR update; full-run schedule."""
    import torch
    device = cuda(args); root = find_root(args.dataset_root); out = fresh(args.output_dir)
    save_config(out, args); meter = Meter(out, device)
    value = symbols(args.official_root, args.backbone)
    transform, _ = value['get_data_transforms'](448, 392)
    for c in selected(args):
        all_items = image_records(root, c, 'train'); items = all_items[:32]
        if len(items) != 32:
            raise ValueError('smoke requires 32 distinct normal training images')
        value['setup_seed'](1)
        with meter.measure('initialization', c):
            model, trainable = build(value, args.official_root, device)
        initial = [p.detach().cpu().clone() for p in trainable.parameters()]
        learning_rates = []
        smoke_symbols = dict(value)
        class ObservedAdamW(value['StableAdamW']):
            def step(self, *step_args, **step_kwargs):
                learning_rates.append(float(self.param_groups[0]['lr']))
                return super().step(*step_args, **step_kwargs)
        smoke_symbols['StableAdamW'] = ObservedAdamW
        def full_schedule(optimizer, **kwargs):
            kwargs['total_iters'] = 200*(len(all_items)//16)
            return value['WarmCosineScheduler'](optimizer, **kwargs)
        smoke_symbols['WarmCosineScheduler'] = full_schedule
        with meter.measure('two_training_steps', c):
            history = external.train_model(model, trainable, items, transform, smoke_symbols, device, 1, 16, args.workers)
        changed = 0
        for previous, parameter in zip(initial, trainable.parameters()):
            current = parameter.detach().cpu()
            if not torch.isfinite(current).all():
                raise FloatingPointError('nonfinite trainable parameters after smoke')
            changed += int(not torch.equal(previous, current))
        del initial
        if len(learning_rates) != 2 or learning_rates[0] != 0 or learning_rates[1] <= 0 or not changed:
            raise RuntimeError('smoke did not verify a nonzero-LR parameter update')
        with meter.measure('normal_only_prediction', c):
            before = list(predict_batches(model, items[:2], transform, value, device, args.workers))
        dest = out/c; dest.mkdir(); write_csv(dest/'smoke_loss.csv', history)
        save_checkpoint(dest/'smoke.pt', model, c, len(all_items), smoke=True)
        del model, trainable; gc.collect(); torch.cuda.empty_cache()
        with meter.measure('reload_and_prediction', c):
            model, trainable = build(value, args.official_root, device)
            state = torch.load(dest/'smoke.pt', map_location='cpu', weights_only=False)
            restore_model(model, state); del state
            after = list(predict_batches(model, items[:2], transform, value, device, args.workers))
        np.testing.assert_array_equal(before[0][1], after[0][1])
        np.testing.assert_array_equal(before[0][2], after[0][2])
        write_json(dest/'complete.json', dict(stage='smoke', reload_prediction_identical=True,
                   training_steps=2, learning_rates=learning_rates, changed_parameter_tensors=changed,
                   nonzero_lr_update_verified=True,
                   schedule_total_iters=200*(len(all_items)//16), formal_result=False))
        del model, trainable; gc.collect(); torch.cuda.empty_cache()
    write_json(out/'complete.json', dict(stage='smoke', formal_result=False, categories=selected(args)))


def metrics_smoke(args):
    """Synthetic arrays only; exercise the existing CUDA metric path against its CPU path."""
    import torch
    from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import evaluate_fast, METRIC_NAMES
    from DINOv3.MADEqual.hard_sample_discrimination.scoring import fixed_fpr_diagnostics
    device = cuda(args); out = fresh(args.output_dir)
    save_config(out, args); meter = Meter(out, device)
    rng = np.random.default_rng(1)
    masks = [np.zeros((256, 256), bool) for _ in range(4)]
    masks[1][60:92, 90:118] = True
    masks[1][150:154, 180:184] = True  # Small region in this evaluation coordinate system.
    masks[2][30:70, 40:70] = True
    labels = [0, 1, 1, 1]  # Includes an anomalous image with an empty mask.
    maps = [rng.random(m.shape).astype(np.float32) + .35*m.astype(np.float32) for m in masks]
    with meter.measure('synthetic_cuda_metrics', ''):
        gpu = evaluate_fast(labels, masks, maps, device=device, allow_cpu_fallback=False)
    with meter.measure('synthetic_cpu_reference', ''):
        cpu = evaluate_fast(labels, masks, maps, device=torch.device('cpu'), allow_cpu_fallback=True)
    for key in METRIC_NAMES:
        np.testing.assert_allclose(gpu[key], cpu[key], rtol=1e-5, atol=1e-6, err_msg=key)
    with meter.measure('synthetic_fixed_fpr', ''):
        operating = fixed_fpr_diagnostics(maps, masks)
    if any(r['actual_fpr'] > r['fpr_cap'] or r['small_region_count'] != 1 for r in operating):
        raise RuntimeError('synthetic fixed-FPR contract failed')
    write_json(out/'complete.json', dict(stage='metrics-smoke', formal_result=False,
               inputs='synthetic_only_no_BTAD_images_or_labels', cuda_cpu_agree=True,
               cuda_metrics=gpu, cpu_reference=cpu, fixed_fpr=operating))


def parser():
    p = argparse.ArgumentParser(description=TITLE)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('check','smoke','metrics-smoke','train','predict','evaluate'):
        q = sub.add_parser(name)
        if name != 'metrics-smoke': q.add_argument('--dataset-root', required=True)
        q.add_argument('--output-dir', required=True, help='must not already exist')
        q.add_argument('--category', choices=('all',*CATEGORIES), default='all')
        q.add_argument('--official-root', default='external_baselines/INP-Former')
        q.add_argument('--device', default='cuda:0')
        q.add_argument('--workers', type=int, default=4)
        if name in ('smoke','train','predict'):
            q.add_argument('--backbone', required=True)
        if name == 'predict': q.add_argument('--checkpoint-root', required=True)
        if name == 'train': q.add_argument('--resume', help='resume_latest.pt from an interrupted category; use a NEW output directory')
        if name == 'evaluate': q.add_argument('--prediction-root', required=True)
    return p


def main():
    args = parser().parse_args()
    if args.workers < 0: raise ValueError('workers must be nonnegative')
    globals()[args.command.replace('-', '_')](args)


if __name__ == '__main__':
    main()
