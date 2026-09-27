"""Existing model/trainer; only prediction-only plumbing is adapted here."""
import contextlib
import importlib
from pathlib import Path
import sys
import tempfile

import numpy as np

from external_baselines.inpformer_external import run as external


def symbols(official_root, backbone):
    # Require local weights before any official import could attempt a download.
    backbone = Path(backbone).resolve()
    if not backbone.is_file():
        raise FileNotFoundError(backbone)
    # Reuse the already audited encoder identity, so a renamed RobustAD detector
    # cannot silently pass upstream load_state_dict(strict=False).
    if external.file_hash(backbone) != external.EXPECTED_BACKBONE_SHA256:
        raise ValueError('requires the audited original DINOv2 reg4 base pretrained weights')
    official_root = Path(official_root).resolve()
    sys.path.insert(0, str(official_root))
    # vit_encoder creates backbones/weights at import time. Confine that empty
    # cache directory to temporary storage, allowing read-only Kaggle source.
    exports = {
        'dataset': ['get_data_transforms'],
        'utils': ['setup_seed', 'WarmCosineScheduler', 'global_cosine_hm_adaptive',
                  'cal_anomaly_maps', 'get_gaussian_kernel'],
        'models.uad': ['INP_Former'],
        'models.vision_transformer': ['Mlp', 'Aggregation_Block', 'Prototype_Block'],
        'optimizers': ['StableAdamW'],
        'models.vit_encoder': [],
    }
    value = {}
    with tempfile.TemporaryDirectory(prefix='inpformer_import_') as scratch:
        with external.working_directory(Path(scratch)):
            for name, names in exports.items():
                module = importlib.import_module(name)
                try:
                    Path(module.__file__).resolve().relative_to(official_root)
                except ValueError:
                    raise RuntimeError('official import name collision: '+name)
                value.update({key:getattr(module,key) for key in names})
                if name == 'models.vit_encoder': value['vit_encoder'] = module
    encoder = value['vit_encoder']
    original = encoder.download_cached_file

    def local_only(url, *args, **kwargs):
        if url.rsplit('/', 1)[-1] != external.BACKBONE_FILENAME:
            raise ValueError('unexpected backbone request: '+url)
        return str(backbone)

    # Scoped file resolver, not a model or checkpoint conversion.
    @contextlib.contextmanager
    def build_context():
        encoder.download_cached_file = local_only
        try:
            yield
        finally:
            encoder.download_cached_file = original
    value['local_backbone_context'] = build_context
    return value


def build(value, official_root, device):
    with value['local_backbone_context']():
        return external.build_model(value, Path(official_root).resolve(), device)


def predict_batches(model, records, transform, value, device, workers=4):
    """Prediction block from external.score_records, without its GT dataset/path."""
    import torch
    import torch.nn.functional as F
    loader = torch.utils.data.DataLoader(
        external.TrainDataset(records, transform), batch_size=16, shuffle=False,
        **external.data_loader_options(workers))
    kernel = value['get_gaussian_kernel'](kernel_size=5, sigma=4).to(device)
    model.eval()
    offset = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            en, de = model(images)[:2]
            maps, _ = value['cal_anomaly_maps'](en, de, images.shape[-1])
            maps = kernel(F.interpolate(maps, size=256, mode='bilinear', align_corners=False))
            flat = maps.flatten(1)
            scores = torch.sort(flat, dim=1, descending=True)[0][:, :int(flat.shape[1]*.01)].mean(1)
            maps, scores = maps[:, 0].cpu().numpy(), scores.cpu().numpy()
            if maps.dtype != np.float32 or not np.isfinite(maps).all() or not np.isfinite(scores).all():
                raise FloatingPointError('invalid FP32 predictions')
            yield offset, maps, scores
            offset += len(maps)
