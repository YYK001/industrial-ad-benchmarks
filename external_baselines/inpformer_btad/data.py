"""Reuse BTAD listing/pairing; keep image-only execution separate from annotations."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from DINOv3.MADEqual.btad_validation.dataset import (
    CATEGORIES, find_root, images, records, mask_info,
)

SPATIAL = dict(resize_hw=[448, 448], crop_xyxy=[28, 28, 420, 420],
               evaluation_hw=[256, 256], field_of_view='official_center_crop',
               image_interpolation='official torchvision PIL bilinear',
               mask_interpolation='nearest, binary before and after resampling')


@dataclass(frozen=True)
class ImageRecord:
    path: str


def image_records(root, category, split):
    """Never reads annotations, even for inference. Same order as BTAD records()."""
    if category not in CATEGORIES or split not in ('train', 'test'):
        raise ValueError('invalid category/split')
    kinds = ('ok',) if split == 'train' else ('ok', 'ko')
    result = [ImageRecord(str(p.resolve())) for k in kinds for p in images(Path(root)/category/split/k)]
    if not result:
        raise ValueError('empty image split')
    return result


def geometry(path):
    with Image.open(path) as im:
        w, h = im.size
    return dict(SPATIAL, original_hw=[h, w],
                original_crop_xyxy=[w/16, h/16, w*15/16, h*15/16])


def load_gt(record):
    """GT in the observed crop only; no extrapolation to the original image."""
    if not record.mask_path:
        return np.zeros((256, 256), bool), dict(raw_empty=False, cropped_empty=False, evaluation_empty=False)
    with Image.open(record.path) as im:
        hw = (im.height, im.width)
    info = mask_info(record.mask_path, hw)
    with Image.open(record.mask_path) as im:
        binary = Image.fromarray((np.asarray(im.convert('L')) > 0).astype(np.uint8)*255)
    nearest = Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST
    crop = binary.resize((448, 448), nearest).crop((28, 28, 420, 420))
    result = np.asarray(crop.resize((256, 256), nearest)) > 0
    return result, dict(raw_empty=info['empty'], cropped_empty=not bool(np.asarray(crop).any()),
                        evaluation_empty=not bool(result.any()))
