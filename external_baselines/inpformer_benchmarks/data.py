"""Official dataset partitions, image-only fitting/scoring, independent binary GT loading."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from DINOv3.relation_reliability.datasets import Record, mvtec_ad_records, MVTEC_AD_CATEGORIES
from DINOv3.MADEqual.visa_task_decoupled.dataset import VISA_CATEGORIES, VISA_FIELDS, visa_one_class_records, _relative_path
from external_baselines.inpformer_btad.data import ImageRecord, SPATIAL, geometry, images

CATEGORIES = {'mvtec': tuple(MVTEC_AD_CATEGORIES), 'visa': VISA_CATEGORIES}


def discover(input_root):
    """Read directory names/CSV existence, never unpack or copy image data."""
    root = Path(input_root).resolve()
    candidates = {k:set() for k in CATEGORIES}
    for dataset, first in (('mvtec','bottle'), ('visa','candle')):
        for folder in root.rglob(first):
            candidate = folder.parent
            if folder.is_dir() and all((candidate/c/'train/good').is_dir() for c in CATEGORIES[dataset]):
                candidates[dataset].add(candidate)
    for split in root.rglob('1cls.csv'):
        if split.parent.name == 'split_csv': candidates['visa'].add(split.parent.parent)
    return {k:sorted(str(p) for p in v) for k,v in candidates.items()}


class Adapter:
    def __init__(self, dataset):
        if dataset not in CATEGORIES: raise ValueError(dataset)
        self.dataset = dataset
        self.CATEGORIES = CATEGORIES[dataset]
        self.TITLE = 'INP-Former官方单类full-shot配置的'+('MVTec AD' if dataset=='mvtec' else 'VisA')+'复评'
        self.PROTOCOL = f'inpformer_{dataset}_single_class_fullshot_seed1_crop_v1'

    def find_root(self, path):
        p=Path(path).expanduser().resolve()
        variants = (p,p/'mvtec',p/'mvtec_ad',p/'mvtec_anomaly_detection') if self.dataset=='mvtec' else (
            p,p/'VisA',p/'Visa',p/'VisA_20220922',p/'1cls',p/'VisA_pytorch/1cls')
        found=[]
        for candidate in variants:
            raw = self.dataset=='visa' and (candidate/'split_csv/1cls.csv').is_file()
            prepared = all((candidate/c/'train/good').is_dir() for c in self.CATEGORIES)
            if raw or prepared: found.append(candidate)
        if len(found)!=1: raise ValueError(f'Expected one {self.dataset} root; candidates={found}; supplied={p}')
        return found[0]

    def layout(self, root):
        return 'official_1cls_csv' if self.dataset=='visa' and (Path(root)/'split_csv/1cls.csv').is_file() else 'train_good_test_defect_folders'

    def image_records(self, root, category, split):
        root=Path(root)
        if category not in self.CATEGORIES or split not in ('train','test'): raise ValueError('category/split')
        if self.layout(root)=='official_1cls_csv':
            paths=[]
            with (root/'split_csv/1cls.csv').open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if tuple(reader.fieldnames or ())!=VISA_FIELDS: raise ValueError('unexpected VisA split CSV fields')
                for row in reader:
                    if row['object'].strip()!=category or row['split'].strip()!=split: continue
                    # Only training-row normal status is consulted. Inference never uses label/mask.
                    if split=='train' and row['label'].strip()!='normal': raise ValueError('non-normal VisA training row')
                    relative=_relative_path(row['image'],field='image')
                    if relative.parts[0]!=category: raise ValueError('image/category mismatch')
                    paths.append(root.joinpath(*relative.parts).resolve())
        elif split=='train':
            if self.dataset=='visa':
                for folder in (root/category/'train').iterdir():
                    if folder.is_dir() and folder.name!='good' and images(folder):
                        raise ValueError('prepared VisA must be normal-only 1cls, not 2cls/few-shot')
            paths=images(root/category/'train/good')
        else:
            folder=root/category/'test'
            if not folder.is_dir(): raise FileNotFoundError(folder)
            paths=[p for d in sorted(folder.iterdir()) if d.is_dir() for p in images(d)]
        if not paths or len(set(paths))!=len(paths): raise ValueError('empty or duplicate images')
        for p in paths:
            if not p.is_file(): raise FileNotFoundError(p)
        return [ImageRecord(str(p.resolve())) for p in paths]

    def records(self, path, categories=None, inspect_masks=False):
        root=self.find_root(path); categories=tuple(categories or self.CATEGORIES)
        if not set(categories)<=set(self.CATEGORIES): raise ValueError('unknown categories')
        if self.dataset=='mvtec':
            train,test=mvtec_ad_records(root,categories)
        elif self.layout(root)=='official_1cls_csv':
            _,all_train,all_test=visa_one_class_records(root)
            train={c:all_train[c] for c in categories}; test={c:all_test[c] for c in categories}
        else:
            train={}; test={}
            for c in categories:
                train[c]=[Record(r.path,0,None,c,'visa','source','good','train','not_evaluation')
                          for r in self.image_records(root,c,'train')]
                test[c]=[]
                for r in self.image_records(root,c,'test'):
                    p=Path(r.path); kind=p.parent.name; mask=None
                    if kind!='good':
                        options=[q for q in images(root/c/'ground_truth'/kind) if q.stem in (p.stem,p.stem+'_mask')]
                        if len(options)!=1: raise ValueError(f'expected one mask for {p}; got {options}')
                        mask=str(options[0])
                    test[c].append(Record(r.path,int(kind!='good'),mask,c,'visa','test',kind,'evaluation','pixel'))
        for c in categories:
            if {r.label for r in test[c]}!={0,1}: raise ValueError(f'incomplete test labels: {c}')
            if set(r.path for r in train[c])&set(r.path for r in test[c]): raise ValueError('train/test overlap')
            if inspect_masks:
                for r in test[c]: self.load_gt(r)
        # Shared runner builds its own full-shot counts/manifest, no calibration roles.
        return root,train,test,[],[]

    @staticmethod
    def load_gt(record):
        if not record.mask_path:
            if record.label: raise ValueError('anomaly without mask')
            return np.zeros((256,256),bool),dict(raw_empty=False,cropped_empty=False,evaluation_empty=False)
        with Image.open(record.path) as im: size=im.size
        with Image.open(record.mask_path) as im:
            if im.size!=size: raise ValueError(f'image/mask geometry mismatch: {record.path}')
            # VisA raw masks may encode defect IDs >1. Official preparation maps all nonzero IDs to 255.
            raw=np.asarray(im)
            if raw.ndim!=2: raise ValueError(f'expected scalar mask IDs: {record.mask_path}')
            binary=raw>0
        nearest=Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST
        crop=Image.fromarray(binary.astype(np.uint8)*255).resize((448,448),nearest).crop((28,28,420,420))
        gt=np.asarray(crop.resize((256,256),nearest))>0
        return gt,dict(raw_empty=not bool(binary.any()),cropped_empty=not bool(np.asarray(crop).any()),evaluation_empty=not bool(gt.any()))
