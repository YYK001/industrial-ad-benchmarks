"""CPU contract tests. AST loading isolates official functions from unavailable optional imports.

No full DINOv2 model, pretrained weights, CUDA resource claim or formal metrics here.
"""
import ast
import copy
import importlib.util
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from . import data, model, run

OFFICIAL = Path(__file__).resolve().parents[1]/'INP-Former'
torch.set_num_threads(1)


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


@pytest.fixture
def official():
    ns = dict(torch=torch, F=F, np=np, math=math, partial=partial,
              _LRScheduler=torch.optim.lr_scheduler._LRScheduler, transforms=transforms)
    definitions(OFFICIAL/'utils.py', ['modify_grad_v2','global_cosine_hm_adaptive',
                'WarmCosineScheduler','cal_anomaly_maps','get_gaussian_kernel'], ns)
    definitions(OFFICIAL/'dataset.py', ['get_data_transforms'], ns)
    spec = importlib.util.spec_from_file_location('local_official_optimizer', OFFICIAL/'optimizers/StableAdamW.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    ns['StableAdamW'] = module.StableAdamW
    return ns


@pytest.fixture
def btad(tmp_path):
    root = tmp_path/'BTAD'
    for c in data.CATEGORIES:
        for folder, n in [('train/ok',16), ('test/ok',1), ('test/ko',2)]:
            dest = root/c/folder; dest.mkdir(parents=True)
            for i in range(n):
                Image.fromarray(np.full((64,96), 40+i, np.uint8)).save(dest/f'{i:03d}.bmp')
        dest = root/c/'ground_truth/ko'; dest.mkdir(parents=True)
        mask = np.zeros((64,96), np.uint8)
        mask[16:32,24:48] = 1 if c=='01' else 255
        Image.fromarray(mask).save(dest/'000.png')
        Image.fromarray(np.zeros_like(mask)).save(dest/'001.png')
    return root


def test_listing_pairing_empty_and_label_free(btad):
    root, train, test, _, _ = data.records(btad, inspect_masks=True)
    for c in data.CATEGORIES:
        assert len(data.image_records(root,c,'train')) == len(train[c]) == 16
        assert [r.path for r in data.image_records(root,c,'test')] == [r.path for r in test[c]]
        r = test[c][-1]
        assert r.label == 1 and Path(r.path).suffix == '.bmp' and Path(r.mask_path).suffix == '.png'
        gt, info = data.load_gt(r)
        assert not gt.any() and info['raw_empty'] and info['cropped_empty']
    # Normal-only listing works with the entire test and annotation directories absent.
    (root/'01/test').rename(root/'01/hidden_test')
    (root/'01/ground_truth').rename(root/'01/hidden_gt')
    assert len(data.image_records(root,'01','train')) == 16


def test_crop_geometry_and_binary_encodings(btad, official):
    _, _, test, _, _ = data.records(btad)
    gt1, _ = data.load_gt(test['01'][1]); gt2, _ = data.load_gt(test['02'][1])
    np.testing.assert_array_equal(gt1, gt2)
    assert gt1.dtype == bool
    # Same geometric object in RGB image and GT; compare interior after bilinear image transform.
    r = test['01'][1]
    binary = np.asarray(Image.open(r.mask_path)) > 0
    Image.fromarray(binary.astype(np.uint8)*255).save(r.path)
    tf, _ = official['get_data_transforms'](448,392)
    image, _ = model.external.TrainDataset([data.ImageRecord(r.path)],tf)[0]
    restored = image[0]*.229+.485
    image_mask = F.interpolate(restored[None,None],size=256,mode='bilinear',align_corners=False)[0,0].numpy()>.5
    assert np.mean(image_mask==gt1) > .995
    assert data.geometry(r.path)['original_crop_xyxy'] == [6.,4.,90.,60.]
    border = np.zeros((64,96),np.uint8); border[:2,:]=255
    Image.fromarray(border).save(r.mask_path)
    cropped, info = data.load_gt(r)
    assert not info['raw_empty'] and info['cropped_empty'] and not cropped.any()


class Tiny(torch.nn.Module):
    """Only exercises trainer/scoring contracts; never used in runtime adapter."""
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Conv2d(3,3,1)

    def forward(self, image):
        en = F.adaptive_avg_pool2d(image,(4,4)).detach()
        de = self.decoder(en)
        return [en], [de], de.square().mean()


def test_external_training_matches_official_executable_block(btad, official):
    torch.manual_seed(1); a=Tiny(); b=copy.deepcopy(a)
    tf,_ = official['get_data_transforms'](448,392)
    items = data.image_records(btad,'01','train')
    # Use two updates so the zero initial warmup LR cannot hide an optimizer mismatch.
    torch.manual_seed(1)
    history = model.external.train_model(a,a,items,tf,official,'cpu',2,16,0)
    source = ast.parse((OFFICIAL/'INP_Former_Single_Class.py').read_text())
    main = next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    branch = next(n for n in main.body if isinstance(n,ast.If) and
                  isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Attribute) and n.test.left.attr=='phase')
    start = next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Assign) and
                 isinstance(n.targets[0],ast.Name) and n.targets[0].id=='optimizer')
    end = next(i for i,n in enumerate(branch.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
    dataset = model.external.TrainDataset(items,tf)
    ns = dict(official, nn=torch.nn, model=b, trainable=b, device='cpu',
              args=SimpleNamespace(total_epochs=2), train_data=dataset,
              train_dataloader=torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True,drop_last=True,
                          **model.external.data_loader_options(0)),
              tqdm=lambda loader,**kw: loader, print_fn=lambda *args: None)
    torch.manual_seed(1)
    exec(compile(ast.Module(body=branch.body[start:end+1],type_ignores=[]),'official_training_block','exec'), ns)
    for x,y in zip(a.parameters(),b.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert history[-1]['loss'] == float(np.mean(ns['loss_list']))
    # Exact scheduler and optimizer options come from the executed official block.
    assert ns['lr_scheduler'].total_iters == 2
    assert ns['optimizer'].param_groups[0]['eps'] == 1e-10


def test_reload_and_annotation_independence(btad,official,tmp_path):
    torch.manual_seed(1); net=Tiny(); initial=copy.deepcopy(net.state_dict())
    tf,_ = official['get_data_transforms'](448,392)
    items=data.image_records(btad,'01','train')
    torch.manual_seed(1)
    model.external.train_model(net,net,items,tf,official,'cpu',2,16,0)
    test_items=data.image_records(btad,'01','test')
    before=list(model.predict_batches(net,test_items,tf,official,'cpu',0))[0]
    path=tmp_path/'last.pt'; run.save_checkpoint(path,net,'01',16)
    replacement=Tiny(); run.load_checkpoint(path,replacement,'01')
    after=list(model.predict_batches(replacement,test_items,tf,official,'cpu',0))[0]
    np.testing.assert_array_equal(before[1],after[1]); np.testing.assert_array_equal(before[2],after[2])
    # Change labels independently of image identities, and make every mask unreadable.
    from dataclasses import replace
    _,_,labelled,_,_=data.records(btad)
    flipped=[replace(r,label=1-r.label,mask_path='DO_NOT_READ') for r in labelled['01']]
    poisoned=[data.ImageRecord(r.path) for r in flipped]
    for mask in (btad/'01/ground_truth/ko').iterdir(): mask.write_bytes(b'not a mask')
    again=list(model.predict_batches(replacement,poisoned,tf,official,'cpu',0))[0]
    np.testing.assert_array_equal(before[1],again[1]); np.testing.assert_array_equal(before[2],again[2])
    replacement.load_state_dict(initial); torch.manual_seed(1)
    model.external.train_model(replacement,replacement,items,tf,official,'cpu',2,16,0)
    for x,y in zip(net.parameters(),replacement.parameters()): torch.testing.assert_close(x,y,rtol=0,atol=0)
    run.save_checkpoint(path,net,'01',16,smoke=True)
    with pytest.raises(ValueError): run.load_checkpoint(path,replacement,'01')


def test_existing_metric_backend_empty_masks():
    from DINOv3.MADEqual.mvtec_broad6_compose2.metrics import evaluate_fast
    from DINOv3.MADEqual.hard_sample_discrimination.scoring import fixed_fpr_diagnostics
    masks=[np.zeros((32,32),bool) for _ in range(3)]; masks[1][10:16,10:16]=True
    rng=np.random.default_rng(1)
    maps=[rng.random((32,32)).astype(np.float32)+m.astype(np.float32) for m in masks]
    result=evaluate_fast([0,1,1],masks,maps,device=torch.device('cpu'),allow_cpu_fallback=True)
    assert result['aupro_threshold_count']==200 and np.isfinite(result['AUPRO_at_0p3'])
    rows=fixed_fpr_diagnostics(maps,masks)
    assert [r['fpr_cap'] for r in rows]==[.01,.05]


def test_missing_or_detector_weights_rejected_before_optional_imports(tmp_path):
    with pytest.raises(FileNotFoundError): model.symbols(OFFICIAL,tmp_path/'missing.pth')
    path=tmp_path/'dinov2_vitb14_reg4_pretrain.pth'
    torch.save(dict(model=Tiny().state_dict()),path)
    with pytest.raises(ValueError,match='DINOv2'): model.symbols(OFFICIAL,path)


def test_prediction_matches_existing_score_records(btad, official):
    _,_,test,_,_=data.records(btad)
    tf, gt_tf=official['get_data_transforms'](448,392)
    torch.manual_seed(1); net=Tiny()
    expected=model.external.score_records(net,test['01'],tf,gt_tf,official,'cpu',16,0)
    actual=list(model.predict_batches(net,data.image_records(btad,'01','test'),tf,official,'cpu',0))[0]
    np.testing.assert_array_equal(expected[0],actual[1])
    np.testing.assert_array_equal(expected[4],actual[2])


def test_evaluate_artifacts_macro_and_official_image_scores(btad,tmp_path,monkeypatch):
    """Synthetic persisted predictions; CPU fallback is injected only by this test."""
    import json
    from sklearn.metrics import roc_auc_score,average_precision_score
    from DINOv3.MADEqual.mvtec_broad6_compose2 import metrics
    real=metrics.evaluate_fast
    def cpu_only(*args,**kwargs):
        assert kwargs['allow_cpu_fallback'] is False  # production must request CUDA
        kwargs['allow_cpu_fallback']=True
        return real(*args,**kwargs)
    monkeypatch.setattr(metrics,'evaluate_fast',cpu_only)
    monkeypatch.setattr(run,'cuda',lambda args:torch.device('cpu'))
    prediction=tmp_path/'predictions'; prediction.mkdir()
    for c in data.CATEGORIES:
        dest=prediction/c; dest.mkdir(); rows=[]
        rng=np.random.default_rng(3)
        for i,record in enumerate(data.image_records(btad,c,'test')):
            np.save(dest/f'{i:04d}.npy',rng.random((256,256)).astype(np.float32))
            rows.append(dict(relative_path=Path(record.path).relative_to(btad).as_posix(),
                image_score=[.9,.8,.1][i],prediction=f'{i:04d}.npy',geometry=data.geometry(record.path)))
        run.write_json(dest/'predictions.json',rows)
        run.write_json(dest/'complete.json',dict(protocol=run.PROTOCOL,checkpoint_epoch=200,
                       spatial=data.SPATIAL,train_count=16))
    out=tmp_path/'evaluation'
    args=run.parser().parse_args(['evaluate','--dataset-root',str(btad),'--prediction-root',str(prediction),
                                 '--output-dir',str(out),'--official-root',str(OFFICIAL)])
    run.evaluate(args)
    result=json.loads((out/'complete.json').read_text())['macro'][0]
    assert result['category_count']==3
    assert result['image_AUROC']==roc_auc_score([0,1,1],[.9,.8,.1])
    assert result['image_AUPR']==average_precision_score([0,1,1],[.9,.8,.1])
    assert (out/'fixed_fpr_macro.csv').is_file() and (out/'mask_audit.csv').is_file()
    assert (out/'source_snapshot/official/INP_Former_Single_Class.py').is_file()
    with pytest.raises(FileExistsError): run.fresh(out)


def test_two_step_smoke_nonzero_update_and_reload(btad, official, tmp_path, monkeypatch):
    import json
    for i in range(16,32):
        Image.fromarray(np.full((64,96),40+i,np.uint8)).save(btad/'01/train/ok'/f'{i:03d}.bmp')
    monkeypatch.setattr(run,'cuda',lambda args:torch.device('cpu'))
    monkeypatch.setattr(run,'symbols',lambda *args:dict(official,setup_seed=torch.manual_seed))
    def small_build(*args):
        net=Tiny()
        return net,net
    monkeypatch.setattr(run,'build',small_build)
    out=tmp_path/'smoke'
    args=run.parser().parse_args(['smoke','--dataset-root',str(btad),'--backbone','unused',
        '--category','01','--workers','0','--official-root',str(OFFICIAL),'--output-dir',str(out)])
    run.smoke(args)
    state=json.loads((out/'01/complete.json').read_text())
    assert state['training_steps']==2 and state['learning_rates'][0]==0
    assert state['learning_rates'][1]>0 and state['changed_parameter_tensors']>0
    assert state['nonzero_lr_update_verified'] and state['reload_prediction_identical']
    assert state['schedule_total_iters']==400 and not state['formal_result']


def test_metrics_smoke_synthetic_only(tmp_path, monkeypatch):
    import json
    from DINOv3.MADEqual.mvtec_broad6_compose2 import metrics
    real=metrics.evaluate_fast
    calls=[]
    def cpu_test(*args,**kwargs):
        calls.append(kwargs['allow_cpu_fallback'])
        kwargs['allow_cpu_fallback']=True
        return real(*args,**kwargs)
    monkeypatch.setattr(metrics,'evaluate_fast',cpu_test)
    monkeypatch.setattr(run,'cuda',lambda args:torch.device('cpu'))
    out=tmp_path/'metrics_smoke'
    args=run.parser().parse_args(['metrics-smoke','--official-root',str(OFFICIAL),'--output-dir',str(out)])
    run.metrics_smoke(args)
    state=json.loads((out/'complete.json').read_text())
    assert calls==[False,True] and state['cuda_cpu_agree']
    assert not state['formal_result'] and state['fixed_fpr'][0]['small_region_count']==1


@pytest.mark.parametrize('workers', [0, 1])
def test_resume_matches_uninterrupted_training(btad, official, workers):
    import random
    tf,_=official['get_data_transforms'](448,392)
    items=data.image_records(btad,'01','train')
    torch.manual_seed(3); full=Tiny(); interrupted=copy.deepcopy(full)
    torch.manual_seed(7)
    expected=model.external.train_model(full,full,items,tf,official,'cpu',4,16,workers)
    saved={}
    class StopAtBoundary(Exception): pass
    def checkpoint(epoch,history,optimizer,scheduler):
        if epoch==2:
            saved.update(copy.deepcopy(dict(epoch=epoch,history=history,optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=None,model=interrupted.state_dict())))
            raise StopAtBoundary()
    torch.manual_seed(7)
    with pytest.raises(StopAtBoundary):
        model.external.train_model(interrupted,interrupted,items,tf,official,'cpu',4,16,workers,
                                   epoch_callback=checkpoint)
    resumed=Tiny(); resumed.load_state_dict(saved['model'])
    actual=model.external.train_model(resumed,resumed,items,tf,official,'cpu',4,16,workers,resume_state=saved)
    assert expected==actual
    for x,y in zip(full.parameters(),resumed.parameters()):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_checkpoint_every_twenty_epochs(btad, official, tmp_path, monkeypatch):
    monkeypatch.setattr(run,'cuda',lambda args:torch.device('cpu'))
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda device:None)
    monkeypatch.setattr(run,'symbols',lambda *args:dict(official,setup_seed=torch.manual_seed))
    def small_build(*args):
        net=Tiny(); return net,net
    monkeypatch.setattr(run,'build',small_build)
    def simulated_epochs(net,trainable,*args,epoch_callback=None,**kwargs):
        optimizer=official['StableAdamW'](trainable.parameters())
        scheduler=official['WarmCosineScheduler'](optimizer,base_value=.001,final_value=.0001,total_iters=200,warmup_iters=100)
        history=[]
        for epoch in range(1,201):
            history.append(dict(epoch=epoch,loss=1.0))
            epoch_callback(epoch,history,optimizer,scheduler)
        return history
    monkeypatch.setattr(model.external,'train_model',simulated_epochs)
    out=tmp_path/'checkpoint_test'
    args=run.parser().parse_args(['train','--dataset-root',str(btad),'--backbone','unused','--category','01',
        '--workers','0','--official-root',str(OFFICIAL),'--output-dir',str(out)])
    run.train(args)
    dest=out/'01'
    assert sorted(p.name for p in dest.glob('epoch_*.pt'))==[f'epoch_{i:03d}.pt' for i in range(20,201,20)]
    state=torch.load(dest/'resume_latest.pt',weights_only=False)
    assert state['epoch']==200 and len(state['history'])==200 and 'optimizer' in state and 'scheduler' in state
    assert not (dest/'checkpoint.tmp').exists()
    assert run.load_checkpoint(dest/'last.pt',Tiny(),'01')==16
