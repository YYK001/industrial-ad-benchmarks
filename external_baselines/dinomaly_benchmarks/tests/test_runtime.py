"""Small scientific-runtime contract checks; no backbone downloads or formal training."""
import ast
import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(n) is not None for n in ('torch','torchvision','numpy','PIL','timm','scipy'))


@unittest.skipUnless(AVAILABLE,'Scientific environment unavailable; real validation remains pending')
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy as np
        from external_baselines.dinomaly_benchmarks.protocol import OFFICIAL_ROOT
        from external_baselines.dinomaly_benchmarks.official import symbols
        cls.torch,cls.np,cls.root = torch,np,OFFICIAL_ROOT
        cls.module = symbols(OFFICIAL_ROOT,'mvtec')

    def test_RGB_and_mask_match_official_dataset_and_visa_nonzero_ids(self):
        from PIL import Image
        import torch.nn.functional as F
        from external_baselines.dinomaly_benchmarks import data
        torch,np,module = self.torch,self.np,self.module
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir,mask_dir = root/'test'/'crack',root/'ground_truth'/'crack'
            image_dir.mkdir(parents=True);mask_dir.mkdir(parents=True)
            image = (np.arange(67*103*3)%256).reshape(67,103,3).astype(np.uint8)
            mask = np.zeros((67,103),np.uint8);mask[22:25,38:40]=255
            Image.fromarray(image).save(image_dir/'000.png')
            Image.fromarray(mask).save(mask_dir/'000_mask.png')
            transform,gt_transform = module.get_data_transforms(448,392)
            reference = module.MVTecDataset(str(root),transform,gt_transform,'test')
            original,original_gt,_,_ = reference[0]
            record = SimpleNamespace(path=str(image_dir/'000.png'),mask_path=str(mask_dir/'000_mask.png'),label=1)
            adapted = data.Images([record],transform)[0]
            torch.testing.assert_close(adapted,original,rtol=0,atol=0)
            expected = F.interpolate(original_gt[None],size=256,mode='nearest').bool()[0,0].numpy()
            actual,audit = data.gt(record,'mvtec',module)
            np.testing.assert_array_equal(actual,expected)
            mask[mask>0]=7
            Image.fromarray(mask).save(mask_dir/'000_mask.png')
            visa,_ = data.gt(record,'visa',module)
            np.testing.assert_array_equal(visa,expected)
            meta = data.metadata(record.path,root,'candle')
            self.assertEqual(meta['original_hw'],[67,103])
            self.assertEqual(meta['crop_xyxy'],[28,28,420,420])
            self.assertFalse(audit['evaluation_empty'])
            # The anomalous image label survives a crop that discards all GT.
            mask[:]=0;mask[0,0]=7
            Image.fromarray(mask).save(mask_dir/'000_mask.png')
            _,audit = data.gt(record,'visa',module)
            self.assertFalse(audit['raw_empty']);self.assertTrue(audit['cropped_empty'])
            self.assertEqual(record.label,1)

    def test_anomaly_maps_and_top655_match_upstream_evaluation_block(self):
        from external_baselines.dinomaly_benchmarks.official import score
        torch,np = self.torch,self.np
        utils = sys.modules['utils']
        torch.manual_seed(3)
        en = [torch.randn(2,4,28,28),torch.randn(2,4,28,28)]
        de = [x + .2*torch.randn_like(x) for x in en]
        class Model:
            def eval(self): return self
            def __call__(self,x): return en,de
        images = torch.zeros(2,3,392,392)
        actual,scores = score(Model(),images,self.module)
        # Execute unchanged original per-batch evaluation statements, excluding
        # its CUDA timer and full-dataset CPU metrics (including compute_pro).
        tree = ast.parse((self.root/'utils.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluation_batch')
        context = next(n for n in fn.body if isinstance(n,ast.With))
        loop = next(n for n in context.body if isinstance(n,ast.For))
        namespace = dict(vars(utils),img=images,gt=torch.zeros(2,1,392,392),label=torch.tensor([0,1]),
            model=Model(),device=torch.device('cpu'),resize_mask=256,max_ratio=.01,
            gaussian_kernel=utils.get_gaussian_kernel(5,4),gt_list_px=[],pr_list_px=[],gt_list_sp=[],pr_list_sp=[])
        exec(compile(ast.Module(body=loop.body,type_ignores=[]),'official_reference','exec'),namespace)
        torch.testing.assert_close(actual,namespace['pr_list_px'][0][:,0],rtol=0,atol=0)
        torch.testing.assert_close(scores,namespace['pr_list_sp'][0],rtol=0,atol=0)
        self.assertEqual(int(256*256*.01),655)

    def test_resume_replays_mid_epoch_and_boundary_with_official_optimizer_scheduler(self):
        from external_baselines.dinomaly_benchmarks import checkpoint as cp
        torch = self.torch
        def setup():
            self.module.setup_seed(1)
            model = torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.2))
            optimizer = self.module.StableAdamW(model.parameters(),lr=.002,betas=(.9,.999),weight_decay=1e-4,amsgrad=True,eps=1e-8)
            scheduler = self.module.WarmCosineScheduler(optimizer,.002,.0002,5000,100)
            loader = torch.utils.data.DataLoader(torch.arange(32,dtype=torch.float32).reshape(16,2)/32,
                batch_size=4,shuffle=True,drop_last=True,num_workers=0)
            return model,optimizer,scheduler,loader
        def steps(model,opt,sched,stream,n):
            seen=[]
            for _ in range(n):
                batch=stream.next();seen.append(batch.clone())
                loss=model(batch).square().mean();opt.zero_grad();loss.backward();opt.step();sched.step()
            return seen
        for cut in (3,4):
            m,o,s,l=setup();stream=cp.Stream(l,'cpu')
            reference_batches=steps(m,o,s,stream,9)
            reference=copy.deepcopy(m.state_dict())
            m,o,s,l=setup();stream=cp.Stream(l,'cpu');steps(m,o,s,stream,cut)
            payload=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),
                scheduler=copy.deepcopy(s.state_dict()),rng=cp.rng('cpu'),loader_cursor=copy.deepcopy(stream.state))
            # Rebuilding consumes RNG, as a real restart does.
            m,o,s,l=setup();m.load_state_dict(payload['model']);o.load_state_dict(payload['optimizer']);s.load_state_dict(payload['scheduler'])
            stream=cp.Stream(l,'cpu',payload['loader_cursor'],payload['rng'])
            seen=steps(m,o,s,stream,9-cut)
            for a,b in zip(seen,reference_batches[cut:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for k,v in reference.items():torch.testing.assert_close(m.state_dict()[k],v,rtol=0,atol=0)
            self.assertEqual(s.last_epoch,9)
            self.assertGreater(o.param_groups[0]['lr'],0)
        # Check update5000 and scheduler final transition without 5000 model steps.
        state=s.state_dict();state['last_epoch']=4999;s.load_state_dict(state);s.step()
        self.assertEqual(s.last_epoch,5000)
        self.assertEqual(o.param_groups[0]['lr'],.0002)

    def test_compact_save_strict_reload_and_prediction_cache(self):
        from external_baselines.dinomaly_benchmarks import checkpoint as cp
        from external_baselines.patchcore_official_eval.storage import npz_write
        torch,np=self.torch,self.np
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Linear(2,2);self.decoder=torch.nn.Linear(2,1)
            def forward(self,x):return self.decoder(self.encoder(x))
        model=Model();x=torch.randn(3,2);before=model(x).detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'final.pt';cp.save(path,dict(weights=cp.compact(model)))
            state=torch.load(path,map_location='cpu',weights_only=False)
            with torch.no_grad():model.decoder.weight.add_(5)
            cp.reload(model,state['weights']);torch.testing.assert_close(model(x),before,rtol=0,atol=0)
            with self.assertRaises(ValueError):cp.reload(model,{})
            array=np.arange(65536,dtype=np.float32).reshape(256,256)/65536
            path=Path(tmp)/'prediction.npz';score=np.asarray(.25,np.float32)
            npz_write(path,anomaly_map=array,image_score=score,mask=array>.8)
            with np.load(path,allow_pickle=False) as cached:
                np.testing.assert_array_equal(array,cached['anomaly_map'])
                self.assertEqual(float(cached['image_score']),float(score))
                np.testing.assert_array_equal(cached['mask'],array>.8)

    def test_existing_category_macro_NA_and_full_vs_partial_scope(self):
        import csv
        import contextlib
        import io
        import json
        from external_baselines.dinomaly_benchmarks.run import summarize
        from external_baselines.patchcore_official_eval.storage import json_write
        from external_baselines.dinomaly_benchmarks.protocol import CATEGORIES, CONFIG
        metrics=[dict(category=c,image_AUROC=.5+i*.01,image_AP=.6,pixel_AUROC=.7,pixel_AP=.8,AUPRO_at_0p3=.9)
                 for i,c in enumerate(CATEGORIES['visa'])]
        fixed=[dict(category=r['category'],fpr_cap=cap,actual_fpr=cap,defect_pixel_recall=.5,region_mean_coverage=.4,
                    small_region_mean_coverage='N/A' if i==0 else .3,region_count=2,small_region_count=0 if i==0 else 1,
                    small_region_image_count=0 if i==0 else 1) for i,r in enumerate(metrics) for cap in (.01,.05)]
        with tempfile.TemporaryDirectory() as tmp:
            args=SimpleNamespace(output_dir=Path(tmp),dataset='visa',evaluation_name='v1')
            for row in metrics:
                folder=Path(tmp)/'visa'/row['category']/'evaluate'/'v1'
                run=dict(config=CONFIG,dataset='visa',category=row['category'],smoke=False)
                json_write(folder/'complete.json',dict(status='complete',identity=dict(prediction_identity=dict(run=run))))
                json_write(folder/'result.json',dict(metrics={k:v for k,v in row.items() if k!='category'},
                    fixed_fpr=[{k:v for k,v in r.items() if k!='category'} for r in fixed if r['category']==row['category']]))
            with contextlib.redirect_stdout(io.StringIO()):summarize(args,list(CATEGORIES['visa']))
            summary=Path(tmp)/'visa'/'summaries'/'v1'
            saved=(summary/'macro_metrics.csv').read_bytes()
            with (summary/'macro_metrics.csv').open() as f:macro=list(csv.DictReader(f))[0]
            with (summary/'fixed_fpr_macro.csv').open() as f:operating=list(csv.DictReader(f))[0]
            self.assertEqual(macro['scope'],'dataset_macro')
            self.assertAlmostEqual(float(macro['image_AUROC']),.555)
            self.assertEqual(int(operating['small_region_mean_coverage_valid_category_count']),11)
            subset=list(CATEGORIES['visa'])[:2]
            with contextlib.redirect_stdout(io.StringIO()):summarize(args,subset)
            self.assertEqual((summary/'macro_metrics.csv').read_bytes(),saved)
            partial=summary/'selected'/'_'.join(subset)/'complete.json'
            self.assertIn('NOT_full_dataset',json.loads(partial.read_text())['scope'])
            # A missing formal category must fail before a new full macro is written.
            (Path(tmp)/'visa'/CATEGORIES['visa'][-1]/'evaluate'/'v1'/'complete.json').unlink()
            with self.assertRaises(FileNotFoundError):summarize(args,list(CATEGORIES['visa']))


if __name__=='__main__':
    unittest.main()
