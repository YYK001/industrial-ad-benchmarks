"""Existing split adapters; only Dinomaly's pinned official spatial transforms."""
from pathlib import Path
from .protocol import CONFIG


def adapter(dataset):
    from external_baselines.inpformer_benchmarks.data import Adapter
    return Adapter(dataset)  # Listing/pairing ONLY; never Adapter.load_gt().


def image_records(dataset, root, category, split):
    return adapter(dataset).image_records(root, category, split)


def records(dataset, root, categories):
    value, train, test, _, _ = adapter(dataset).records(root, categories, inspect_masks=False)
    return value, train, test


def metadata(path, root, category):
    from PIL import Image
    with Image.open(path) as image:
        w,h = image.size
    return dict(category=category, relative_path=Path(path).relative_to(root).as_posix(),
        original_hw=[h,w], resize_hw=[448,448], input_hw=[392,392], crop_xyxy=[28,28,420,420],
        original_crop_xyxy=[28*w/448,28*h/448,420*w/448,420*h/448], evaluation_hw=[256,256],
        coordinate_rule='official center crop of resized image; no full-frame map or border padding')


class Images:
    def __init__(self, items, transform):
        self.paths = [str(r.path) for r in items]
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert('RGB'))


def loader(items, transform, batch, workers, training=False):
    import torch
    options = dict(num_workers=workers, pin_memory=False)
    if workers:
        options['multiprocessing_context'] = 'spawn'
    return torch.utils.data.DataLoader(Images(items,transform), batch_size=batch,
        shuffle=training, drop_last=training, **options)


def gt(record, dataset, module):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    if not record.mask_path:
        if record.label:
            raise ValueError('Anomalous image without mask')
        return np.zeros((256,256),dtype=bool), dict(raw_empty=False,cropped_empty=False,evaluation_empty=False)
    with Image.open(record.path) as image:
        original_size = image.size
    with Image.open(record.mask_path) as image:
        if image.size != original_size:
            raise ValueError('GT/image original geometry mismatch')
        raw = np.array(image)
        raw_empty = not bool((raw != 0).any())
        if dataset == 'visa':
            if raw.ndim != 2:
                raise ValueError('Expected scalar VisA defect IDs')
            # Exactly the semantic binarization done by official VisA preparation.
            image = Image.fromarray((raw != 0).astype(np.uint8)*255)
        _, transform = module.get_data_transforms(448,392)
        cropped = transform(image)
    value = F.interpolate(cropped[None], size=256, mode='nearest').bool()
    if value.shape[1] > 1:
        value = torch.max(value, dim=1, keepdim=True)[0]
    mask = value[0,0].numpy()
    return mask, dict(raw_empty=raw_empty, cropped_empty=not bool(cropped.bool().any()),
                      evaluation_empty=not bool(mask.any()))
