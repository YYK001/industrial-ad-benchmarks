"""Frozen official configuration; importable without a scientific environment."""
from pathlib import Path
from external_baselines.patchcore_official_eval.protocol import CATEGORIES as EXISTING

OFFICIAL_ROOT = Path(__file__).resolve().parents[1] / 'dinomaly_official'
COMMIT = '1f252be03a918789b19848f0ca37166e7c28dada'
URL = 'https://github.com/guojiajeremy/Dinomaly'
BACKBONE_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth'
BACKBONE_SHA256 = '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
CATEGORIES = {k: EXISTING[k] for k in ('mvtec', 'visa')}
EXPECTED_COUNTS = {'mvtec': (3629, 1725), 'visa': (8659, 2162)}
CONFIG = dict(method='Dinomaly CVPR2025 conventional class-separated',
    official_url=URL, official_commit=COMMIT, seed=1, total_iters=5000,
    batch_size=16, shuffle=True, drop_last=True, encoder='dinov2reg_vit_base_14',
    target_layers=[2,3,4,5,6,7,8,9], fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],
    fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]], model='official ViTill',
    bottleneck='official bMlp drop=0.2', decoder='8 official Block / LinearAttention2',
    mask_neighbor_size=0, optimizer='official StableAdamW', lr=2e-3,
    betas=[.9,.999], weight_decay=1e-4, amsgrad=True, eps=1e-8,
    scheduler='official WarmCosineScheduler', final_value=2e-4, warmup_iters=100,
    loss='official global_cosine_hm_percent', p='min(0.9 * zero_based_update / 1000, 0.9)',
    factor=.1, clip_grad_max_norm=.1, precision='FP32', amp=False,
    final_selection='fixed optimizer update 5000; no test checkpoint selection',
    spatial=dict(resize_hw=[448,448], input_hw=[392,392], crop_xyxy=[28,28,420,420],
        evaluation_hw=[256,256], field_of_view='official center crop, no full-frame stretching/padding',
        image='official PIL bilinear Resize -> ToTensor -> CenterCrop -> ImageNet Normalize',
        gt='PIL bilinear Resize -> CenterCrop -> ToTensor -> nearest256 -> bool (nonzero); channel max',
        visa_ids='original nonzero IDs -> uint8 255 before official GT transform',
        map='official cal_anomaly_maps(392) -> bilinear256 align_corners=False -> Gaussian kernel5 sigma4',
        image_score='official descending sort, int(256*256*0.01)=655, mean'),
    evaluation='official single-class training/inference + project fast CUDA metric reevaluation',
    aupro='existing CUDA200 / FPR0.3 / four-connected', AP='average precision',
    calibration='none; full normal training set')


def select(dataset, names, shard=None):
    values = list(CATEGORIES[dataset]) if names == ['all'] else list(names)
    if not values or len(set(values)) != len(values) or not set(values) <= set(CATEGORIES[dataset]):
        raise ValueError('Use all or unique categories for this dataset')
    if shard:
        if shard not in ('0/2', '1/2'):
            raise ValueError('Dual T4 shards are 0/2 or 1/2')
        values = values[int(shard[0])::2]
    return values
