import csv
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from .data import Adapter,CATEGORIES,discover
from .run import dataset_context,shared
from . import pipeline


def picture(path,mask=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    value=np.full((48,64),60,np.uint8)
    if mask:
        value[:]=0; value[12:24,20:32]=3
    Image.fromarray(value).save(path)


@pytest.fixture
def fixtures(tmp_path):
    mvtec=tmp_path/'mvtec'
    prepared=tmp_path/'VisA_pytorch/1cls'
    raw=tmp_path/'VisA'
    for root,dataset in ((mvtec,'mvtec'),(prepared,'visa')):
        for c in CATEGORIES[dataset]:
            for i in range(32): picture(root/c/'train/good'/f'{i:03d}.png')
            picture(root/c/'test/good/000.png'); picture(root/c/'test/bad/001.png')
            picture(root/c/'ground_truth/bad'/('001_mask.png' if dataset=='mvtec' else '001.png'),mask=True)
    rows=[]
    for c in CATEGORIES['visa']:
        for i in range(32):
            path=f'{c}/Data/Images/Normal/{i:03d}.JPG'; picture(raw/path)
            rows.append([c,'train','normal',path,''])
        normal=f'{c}/Data/Images/Normal/100.JPG'; picture(raw/normal)
        anomaly=f'{c}/Data/Images/Anomaly/001.JPG'; picture(raw/anomaly)
        mask=f'{c}/Data/Masks/Anomaly/001.png'; picture(raw/mask,mask=True)
        rows.extend([[c,'test','normal',normal,''],[c,'test','anomaly',anomaly,mask]])
    (raw/'split_csv').mkdir()
    with (raw/'split_csv/1cls.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['object','split','label','image','mask']); w.writerows(rows)
    return mvtec,prepared,raw


@pytest.mark.parametrize('dataset,index', [('mvtec',0),('visa',1),('visa',2)])
def test_partition_gt_and_image_order(fixtures,dataset,index):
    adapter=Adapter(dataset); root=fixtures[index]
    actual,train,test,_,_=adapter.records(root,inspect_masks=True)
    assert actual==root and len(train)==len(CATEGORIES[dataset])
    for c in train:
        assert len(train[c])==32 and all(r.label==0 for r in train[c])
        assert [r.path for r in test[c]]==[r.path for r in adapter.image_records(root,c,'test')]
        anomaly=next(r for r in test[c] if r.label)
        gt,info=adapter.load_gt(anomaly)
        assert gt.dtype==bool and gt.shape==(256,256) and gt.any() and not info['raw_empty']
        Image.fromarray(np.zeros((48,64),np.uint8)).save(anomaly.mask_path)
        gt,info=adapter.load_gt(anomaly)
        assert info['raw_empty'] and anomaly.label==1 and not gt.any()


def test_raw_visa_test_metadata_not_used_for_training_or_prediction(fixtures):
    root=fixtures[2]; adapter=Adapter('visa'); c='candle'
    before_train=adapter.image_records(root,c,'train'); before_test=adapter.image_records(root,c,'test')
    path=root/'split_csv/1cls.csv'
    with path.open() as f: rows=list(csv.DictReader(f))
    for row in rows:
        if row['split']=='test': row['label']='CHANGED'; row['mask']='DO_NOT_READ'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['object','split','label','image','mask']); w.writeheader(); w.writerows(rows)
    assert adapter.image_records(root,c,'train')==before_train
    assert adapter.image_records(root,c,'test')==before_test
    with pytest.raises(RuntimeError): adapter.records(root)


def test_discovery_and_prepared_rejects_non_normal_training(fixtures):
    detected=discover(fixtures[0].parent)
    assert detected['mvtec']==[str(fixtures[0])]
    assert set(detected['visa'])=={str(fixtures[1]),str(fixtures[2])}
    picture(fixtures[1]/'candle/train/bad/bad.png')
    with pytest.raises(ValueError,match='1cls'): Adapter('visa').image_records(fixtures[1],'candle','train')


def test_compact_last_checkpoint_and_context_restore(tmp_path):
    original_protocol=shared.PROTOCOL
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.encoder=torch.nn.Linear(3,3); self.decoder=torch.nn.Linear(3,3)
    with dataset_context('mvtec'):
        assert len(shared.CATEGORIES)==15 and shared.PROTOCOL!=original_protocol
        model=Small(); path=tmp_path/'last.pt'
        shared.save_checkpoint(path,model,'bottle',32)
        state=torch.load(path,weights_only=False)
        assert 'model' not in state and not any(k.startswith('encoder.') for k in state['model_without_encoder'])
        other=Small(); encoder_before=other.encoder.weight.detach().clone()
        shared.load_checkpoint(path,other,'bottle')
        torch.testing.assert_close(other.decoder.weight,model.decoder.weight,rtol=0,atol=0)
        torch.testing.assert_close(other.encoder.weight,encoder_before,rtol=0,atol=0)
    assert shared.PROTOCOL==original_protocol and len(shared.CATEGORIES)==3
    with dataset_context('visa'):
        with pytest.raises(ValueError): shared.load_checkpoint(path,Small(),'candle')


def test_resume_selection_uses_epoch_and_prunes_only_owned_files(tmp_path):
    folder=tmp_path/'mvtec/bottle'; paths=[]
    for attempt,epoch in [(0,60),(1,40)]:
        path=folder/f'train_{attempt:03d}/bottle/resume_latest.pt'; path.parent.mkdir(parents=True)
        path.write_bytes(b'fixture'); shared.write_json(path.parent/'checkpoint_progress.json',dict(epoch=epoch))
        paths.append(path)
    selected=pipeline.latest_resume(folder,'bottle'); assert selected==paths[0]
    pipeline.prune_resume(folder,'bottle',selected)
    assert paths[0].exists() and not paths[1].exists()
    marker=paths[0].parent.parent/'complete.json'
    shared.write_json(marker,dict(protocol=Adapter('mvtec').PROTOCOL,stage='train',categories=['bottle']))
    (paths[0].parent/'last.pt').write_bytes(b'fixture')
    assert pipeline.completed_attempt(folder,'train','mvtec','bottle')==marker.parent


def test_macro_packaging_and_excluded_weights(tmp_path):
    root=tmp_path/'mvtec'; completed={}
    for i,c in enumerate(('bottle','cable')):
        paths={k:root/c/(k+'_000') for k in ('train','predict','evaluate')}
        for path in paths.values(): path.mkdir(parents=True)
        row=dict(category=c,**{k:.6+i*.2 for k in pipeline.METRICS})
        shared.write_csv(paths['evaluate']/'category_metrics.csv',[row])
        shared.write_csv(paths['evaluate']/'counts.csv',[dict(category=c,train_normal=32,test_count=2)])
        shared.write_csv(paths['evaluate']/'fixed_fpr.csv',[dict(category=c,fpr_cap=cap,
            defect_pixel_recall=.4+i*.2,region_mean_coverage=.3+i*.2,
            small_region_mean_coverage='N/A' if i else .2) for cap in (.01,.05)])
        (paths['train']/'last.pt').write_bytes(b'exclude weights')
        (paths['predict']/'0000.npy').write_bytes(b'exclude predictions')
        snapshot=paths['evaluate']/'source_snapshot'; snapshot.mkdir(); (snapshot/'run.py').write_text('pass\n')
        completed[c]=paths
    summary=root/'summary'; pipeline.aggregate('mvtec',('bottle','cable'),completed,summary)
    macro=pipeline.read_csv(summary/'macro_metrics.csv')[0]
    assert float(macro['image_AUROC'])==pytest.approx(.7) and macro['scope']=='selected_categories_equal_macro'
    fixed=pipeline.read_csv(summary/'fixed_fpr_macro.csv')[0]
    assert float(fixed['small_region_mean_coverage'])==.2 and int(fixed['small_region_mean_coverage_category_count'])==1
    archive=tmp_path/'results.zip'; pipeline.package_results(root,summary,archive)
    with zipfile.ZipFile(archive) as z:
        assert 'summary/report.md' in z.namelist()
        assert sum(n.endswith('.py') for n in z.namelist())==1
        assert not any(n.endswith(('.pt','.npy')) for n in z.namelist())


@pytest.mark.parametrize('dataset,index,count',[('mvtec',0,15),('visa',2,12)])
def test_full_dataset_evaluation_macro_cpu_fixture(fixtures,tmp_path,monkeypatch,dataset,index,count):
    from DINOv3.MADEqual.mvtec_broad6_compose2 import metrics
    from external_baselines.inpformer_btad.data import geometry
    real=metrics.evaluate_fast
    def cpu_test(*args,**kwargs):
        assert kwargs['allow_cpu_fallback'] is False
        kwargs['allow_cpu_fallback']=True
        return real(*args,**kwargs)
    monkeypatch.setattr(metrics,'evaluate_fast',cpu_test)
    monkeypatch.setattr(shared,'cuda',lambda args:torch.device('cpu'))
    root=fixtures[index]; pred=tmp_path/'predictions'; rng=np.random.default_rng(7)
    with dataset_context(dataset) as adapter:
        _,_,test,_,_=adapter.records(root)
        for c,items in test.items():
            dest=pred/c; dest.mkdir(parents=True); entries=[]
            for i,r in enumerate(items):
                pixel=rng.random((256,256)).astype(np.float32)
                np.save(dest/f'{i:04d}.npy',pixel)
                entries.append(dict(relative_path=Path(r.path).relative_to(root).as_posix(),
                    geometry=geometry(r.path),prediction=f'{i:04d}.npy',image_score=float(r.label)))
            shared.write_json(dest/'predictions.json',entries)
            shared.write_json(dest/'complete.json',dict(protocol=adapter.PROTOCOL,checkpoint_epoch=200,
                spatial=shared.SPATIAL,train_count=32))
        out=tmp_path/'evaluation'
        args=shared.parser().parse_args(['evaluate','--dataset-root',str(root),'--prediction-root',str(pred),'--output-dir',str(out)])
        shared.evaluate(args)
        macro=pipeline.read_csv(out/'macro_metrics.csv')[0]
        assert int(macro['category_count'])==count and macro['scope']==f'{dataset}_{count}_category_equal_macro'
        assert float(macro['image_AUROC'])==1 and float(macro['image_AUPR'])==1
        assert len(pipeline.read_csv(out/'category_metrics.csv'))==count
        assert 'BTAD' not in (out/'report.md').read_text(encoding='utf-8')
