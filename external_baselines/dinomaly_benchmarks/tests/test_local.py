import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from zipfile import ZipFile

from external_baselines.dinomaly_benchmarks import official
from external_baselines.dinomaly_benchmarks.protocol import CONFIG, OFFICIAL_ROOT, select, CATEGORIES
from external_baselines.dinomaly_benchmarks.checkpoint import cursor_valid
from external_baselines.dinomaly_benchmarks.run import export, parser


class LocalTests(unittest.TestCase):
    def test_pinned_source_and_shared_official_construction(self):
        state = official.source(OFFICIAL_ROOT)
        self.assertFalse(state['tracked_source_modified'])
        mvtec = official.construction_ast(OFFICIAL_ROOT/'dinomaly_mvtec_sep.py')
        visa = official.construction_ast(OFFICIAL_ROOT/'dinomaly_visa_sep.py')
        self.assertEqual(ast.dump(mvtec), ast.dump(visa))
        compile(mvtec,'upstream_model','exec')
        code = ast.unparse(mvtec)
        for symbol in ('ViTill(', 'bMlp(', 'LinearAttention2', 'StableAdamW(', 'WarmCosineScheduler(', 'trunc_normal_('):
            self.assertIn(symbol,code)
        for dataset in ('mvtec','visa'):
            tree = ast.parse((OFFICIAL_ROOT/f'dinomaly_{dataset}_sep.py').read_text(encoding='utf-8'))
            train = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
            values = {n.targets[0].id: n.value.value for n in train.body if isinstance(n,ast.Assign)
                      and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
            for name, expected in [('total_iters',5000),('batch_size',16),('image_size',448),('crop_size',392)]:
                self.assertEqual(values[name],expected)

    def test_import_collision_rejected_before_scientific_import(self):
        previous = sys.modules.get('models')
        module = ModuleType('models')
        module.__file__ = str(Path(tempfile.gettempdir())/'wrong_method/models.py')
        sys.modules['models'] = module
        try:
            with self.assertRaisesRegex(RuntimeError,'collision'):
                official.symbols(OFFICIAL_ROOT,'mvtec')
        finally:
            if previous is None:
                del sys.modules['models']
            else:
                sys.modules['models'] = previous

    def test_two_shards_and_datasets_stay_separate(self):
        for dataset,count in [('mvtec',15),('visa',12)]:
            a,b = (select(dataset,['all'],shard) for shard in ('0/2','1/2'))
            self.assertFalse(set(a)&set(b))
            self.assertEqual(set(a+b),set(CATEGORIES[dataset]))
            self.assertEqual(len(a+b),count)
        with self.assertRaises(ValueError):
            select('mvtec',['candle'])
        with self.assertRaises(ValueError):
            select('visa',['candle','candle'])

    def test_resume_cursor_allows_epoch_boundary_but_not_invalid_progress(self):
        cursor_valid(dict(epoch=3,batches=5),5)
        for cursor in (dict(epoch=-1,batches=0),dict(epoch=0,batches=6)):
            with self.assertRaises(ValueError):
                cursor_valid(cursor,5)

    def test_phased_cli_and_light_report_excludes_weights_arrays(self):
        args = parser().parse_args(['evaluate','--dataset','visa'])
        self.assertIsNone(args.backbone)
        self.assertIsNone(args.dataset_root)
        self.assertEqual(CONFIG['total_iters'],5000)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'visa'/'candle'
            root.mkdir(parents=True)
            for name in ('scores.csv','complete.json','final.pt','latest.pt','prediction.npz'):
                (root/name).write_text('{}',encoding='utf-8')
            args = SimpleNamespace(output_dir=Path(tmp),dataset='visa',export_kind='report')
            with contextlib.redirect_stdout(io.StringIO()):
                export(args)
            with ZipFile(Path(tmp)/'Dinomaly_visa_report.zip') as z:
                self.assertEqual(set(z.namelist()),{'visa/candle/scores.csv','visa/candle/complete.json'})


if __name__=='__main__':
    unittest.main()
