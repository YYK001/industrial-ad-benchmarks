"""Pinned INP-Former single-class full-shot evaluation on RobustAD.

This external runner imports the unmodified official implementation and adds
only RobustAD data handling, group-wise reporting, provenance, and the
separately labelled SCRS calibration protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True

try:
    from .dataset_adapter import PIXEL_CATEGORIES, ROBUSTAD_LAYOUT, Record, filter_evaluation, robustad_records, scrs_partition
    from .metrics import aggregate, file_hash, image_metrics, stable_hash, write_csv, write_json
except ImportError:
    from dataset_adapter import PIXEL_CATEGORIES, ROBUSTAD_LAYOUT, Record, filter_evaluation, robustad_records, scrs_partition
    from metrics import aggregate, file_hash, image_metrics, stable_hash, write_csv, write_json


EXPECTED_OFFICIAL_COMMIT = "17d265381d9b323a2ef6e05aab0665a85edebe84"
BACKBONE_FILENAME = "dinov2_vitb14_reg4_pretrain.pth"
EXPECTED_BACKBONE_SHA256 = "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71"
OFFICIAL_SEED = 1
OFFICIAL_CONFIG = {
    "encoder": "dinov2reg_vit_base_14",
    "input_size": 448,
    "crop_size": 392,
    "INP_num": 6,
    "total_epochs": 200,
    "batch_size": 16,
    "num_workers": 4,
    "image_score": "mean_top_1_percent_of_gaussian_smoothed_256_map",
    "gaussian_kernel_size": 5,
    "gaussian_sigma": 4,
    "evaluation_map_size": 256,
}
EXPECTED_PACKAGE_VERSIONS = {
    "matplotlib": "3.2.1",
    "numpy": "1.19.0",
    "opencv-python-headless": "4.6.0.66",
    "pandas": "1.3.5",
    "Pillow": "9.0.1",
    "scikit-image": "0.19.3",
    "scikit-learn": "0.22.2.post1",
    "scipy": "1.4.1",
    "tabulate": "0.9.0",
    "torch": "2.0.0+cu118",
    "torchvision": "0.15.1+cu118",
    "tqdm": "4.64.1",
    "timm": "0.9.12",
    "kornia": "0.7.3",
    "adeval": "1.1.0",
}


def read_git_head_without_git(repo: Path) -> str:
    """Read HEAD metadata without executing a Git command."""
    git_dir = repo / ".git"
    if git_dir.is_file():  # Submodule checkout uses a gitdir pointer file.
        pointer = git_dir.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError("Invalid Git metadata pointer: {}".format(git_dir))
        git_dir = (repo / pointer[8:]).resolve()
    head_file = git_dir / "HEAD"
    if not head_file.is_file():
        raise FileNotFoundError(head_file)
    value = head_file.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value
    reference = value[5:]
    loose = git_dir / reference
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise RuntimeError("Cannot resolve {} from Git metadata".format(reference))


def official_manifest(official_root: Path) -> Dict[str, Any]:
    head = read_git_head_without_git(official_root)
    if head != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError("Official commit mismatch: expected {}, found {}".format(EXPECTED_OFFICIAL_COMMIT, head))
    required = (
        "INP_Former_Single_Class.py", "dataset.py", "utils.py", "requirements.txt",
        "models/uad.py", "models/vit_encoder.py", "models/vision_transformer.py",
        "optimizers/__init__.py", "optimizers/StableAdamW.py",
    )
    hashes = {}
    for relative in required:
        path = official_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = file_hash(path)
    return {"commit": head, "audited_file_sha256": hashes, "audited_files_hash": stable_hash(hashes)}


def external_manifest() -> Dict[str, str]:
    root = Path(__file__).resolve().parent
    return {path.name: file_hash(path) for path in sorted(root.glob("*.py"))}


def package_versions() -> Dict[str, str]:
    versions = {}
    for package in EXPECTED_PACKAGE_VERSIONS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def validate_environment() -> Dict[str, str]:
    if platform.python_version() != "3.8.12":
        raise RuntimeError(
            "Official environment requires Python 3.8.12, found {}".format(platform.python_version())
        )
    actual = package_versions()
    if actual != EXPECTED_PACKAGE_VERSIONS:
        differences = {
            package: {"expected": expected, "actual": actual.get(package)}
            for package, expected in EXPECTED_PACKAGE_VERSIONS.items()
            if actual.get(package) != expected
        }
        raise RuntimeError("Installed packages differ from official requirements: {}".format(differences))
    return actual


def mask_support_after_preprocess(
    path: str, resize_size: int = 448, crop_size: int = 392
) -> Tuple[bool, bool]:
    raw = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0 > 0.5
    mask = Image.open(path).convert("L").resize((resize_size, resize_size), Image.BILINEAR)
    left = (resize_size - crop_size) // 2
    mask = mask.crop((left, left, left + crop_size, left + crop_size))
    processed = np.asarray(mask, dtype=np.float32) / 255.0 > 0.5
    return bool(raw.any()), bool(processed.any())


def audit_dataset(
    train: Dict[str, List[Record]], raw_train: Dict[str, List[Record]], evaluation: Dict[str, List[Record]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    audit_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    for category in train:
        train_hashes: Dict[str, List[str]] = {}
        for record in raw_train[category]:
            digest = file_hash(Path(record.path))
            if record.label == 0:
                train_hashes.setdefault(digest, []).append(record.sample_id)
            row = dict(record.__dict__)
            row["image_sha256"] = digest
            row["protocol_role"] = "native_fullshot_fit" if record.label == 0 else "excluded_non_normal_train_row"
            manifest_rows.append(row)
        groups = sorted({(record.domain, record.shift) for record in evaluation[category]})
        for domain, shift in groups:
            group = [record for record in evaluation[category] if (record.domain, record.shift) == (domain, shift)]
            unreadable_images = 0
            missing_masks = 0
            unreadable_masks = 0
            empty_raw_masks = 0
            empty_processed_masks = 0
            normal_masks = 0
            content_overlap = 0
            for record in group:
                try:
                    with Image.open(record.path) as image:
                        image.verify()
                except Exception:
                    unreadable_images += 1
                digest = file_hash(Path(record.path))
                content_overlap += int(digest in train_hashes)
                row = dict(record.__dict__)
                row["image_sha256"] = digest
                row["protocol_role"] = "evaluation"
                manifest_rows.append(row)
                if record.label == 0:
                    normal_masks += int(record.mask_path is not None)
                elif record.mask_capability == "pixel":
                    if not record.mask_path or not Path(record.mask_path).is_file():
                        missing_masks += 1
                    else:
                        try:
                            raw_nonempty, processed_nonempty = mask_support_after_preprocess(record.mask_path)
                            empty_raw_masks += int(not raw_nonempty)
                            empty_processed_masks += int(not processed_nonempty)
                        except Exception:
                            unreadable_masks += 1
            labels = {record.label for record in group}
            row = {
                "category": category,
                "domain": domain,
                "shift": shift,
                "native_fullshot_normal_fit_images": len(train[category]),
                "raw_training_rows": len(raw_train[category]),
                "normal_images": sum(record.label == 0 for record in group),
                "anomaly_images": sum(record.label == 1 for record in group),
                "mask_capability": group[0].mask_capability,
                "image_metrics_evaluable": labels == {0, 1},
                "pixel_metrics_evaluable": group[0].mask_capability == "pixel",
                "unreadable_images": unreadable_images,
                "missing_anomaly_masks": missing_masks,
                "unreadable_anomaly_masks": unreadable_masks,
                "empty_anomaly_masks_raw": empty_raw_masks,
                "empty_anomaly_masks_after_official_preprocess": empty_processed_masks,
                "normal_rows_with_mask": normal_masks,
                "exact_train_evaluation_content_overlap": content_overlap,
            }
            audit_rows.append(row)
    return audit_rows, manifest_rows


def audit_failures(audit_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    failures = []
    for row in audit_rows:
        if (
            not row["image_metrics_evaluable"]
            or row["unreadable_images"]
            or row["missing_anomaly_masks"]
            or row["unreadable_anomaly_masks"]
            or row["empty_anomaly_masks_raw"]
            or row["normal_rows_with_mask"]
            or row["exact_train_evaluation_content_overlap"]
        ):
            failures.append(row)
    return failures


def audit_warnings(audit_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row for row in audit_rows
        if row["empty_anomaly_masks_after_official_preprocess"]
        and not row["empty_anomaly_masks_raw"]
    ]


def write_audit_outputs(output_dir: Path, audit_rows: List[Dict[str, Any]], manifest_rows: List[Dict[str, Any]]) -> None:
    failures = audit_failures(audit_rows)
    warnings = audit_warnings(audit_rows)
    write_csv(output_dir / "dataset_audit.csv", audit_rows)
    write_csv(output_dir / "dataset_manifest.csv", manifest_rows)
    write_json(
        output_dir / "dataset_audit.json",
        {
            "status": "failed" if failures else ("complete_with_warnings" if warnings else "complete"),
            "evaluation_group_count": len(audit_rows),
            "expected_group_count": 19,
            "groups": audit_rows,
            "failed_groups": failures,
            "warning_groups": warnings,
            "dataset_manifest_sha256": stable_hash(manifest_rows),
        },
    )


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


def import_official(official_root: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(official_root))
    with working_directory(official_root):
        dataset_module = importlib.import_module("dataset")
        utils_module = importlib.import_module("utils")
        vit_encoder = importlib.import_module("models.vit_encoder")
        uad = importlib.import_module("models.uad")
        vision_transformer = importlib.import_module("models.vision_transformer")
        optimizers = importlib.import_module("optimizers")
    return {
        "get_data_transforms": dataset_module.get_data_transforms,
        "setup_seed": utils_module.setup_seed,
        "WarmCosineScheduler": utils_module.WarmCosineScheduler,
        "global_cosine_hm_adaptive": utils_module.global_cosine_hm_adaptive,
        "cal_anomaly_maps": utils_module.cal_anomaly_maps,
        "get_gaussian_kernel": utils_module.get_gaussian_kernel,
        "ader_evaluator": utils_module.ader_evaluator,
        "vit_encoder": vit_encoder,
        "INP_Former": uad.INP_Former,
        "Mlp": vision_transformer.Mlp,
        "Aggregation_Block": vision_transformer.Aggregation_Block,
        "Prototype_Block": vision_transformer.Prototype_Block,
        "StableAdamW": optimizers.StableAdamW,
    }


def build_model(symbols: Dict[str, Any], official_root: Path, device: Any):
    import torch
    import torch.nn as nn
    from torch.nn.init import trunc_normal_

    with working_directory(official_root):
        encoder = symbols["vit_encoder"].load(OFFICIAL_CONFIG["encoder"])
    embed_dim, num_heads = 768, 12
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    bottleneck = nn.ModuleList([symbols["Mlp"](embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    prototypes = nn.ParameterList([nn.Parameter(torch.randn(OFFICIAL_CONFIG["INP_num"], embed_dim))])
    extractor = nn.ModuleList(
        [symbols["Aggregation_Block"](
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )]
    )
    decoder = nn.ModuleList(
        [symbols["Prototype_Block"](
            dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        ) for _ in range(8)]
    )
    model = symbols["INP_Former"](
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=extractor,
        decoder=decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=prototypes,
    ).to(device)
    trainable = nn.ModuleList([bottleneck, decoder, extractor, prototypes])
    for module in trainable.modules():
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    return model, trainable


class TrainDataset:
    def __init__(self, records: Sequence[Record], transform: Any):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image = Image.open(self.records[index].path).convert("RGB")
        return self.transform(image), 0


class EvaluationDataset:
    def __init__(self, records: Sequence[Record], transform: Any, gt_transform: Any, crop_size: int):
        self.records = list(records)
        self.transform = transform
        self.gt_transform = gt_transform
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        image = self.transform(Image.open(record.path).convert("RGB"))
        if record.mask_path:
            mask = self.gt_transform(Image.open(record.mask_path).convert("L"))
        else:
            mask = torch.zeros((1, self.crop_size, self.crop_size), dtype=torch.float32)
        return image, mask, record.label, record.path


def data_loader_options(num_workers: int) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2
    return options


def train_model(
    model: Any, trainable: Any, records: Sequence[Record], transform: Any, symbols: Dict[str, Any],
    device: Any, epochs: int, batch_size: int, num_workers: int,
    resume_state: Optional[Dict[str, Any]] = None, epoch_callback: Any = None,
) -> List[Dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    loader = DataLoader(
        TrainDataset(records, transform), batch_size=batch_size, shuffle=True,
        drop_last=True, **data_loader_options(num_workers),
    )
    if not len(loader):
        raise RuntimeError("No complete training batch: {} images, batch_size={}".format(len(records), batch_size))
    optimizer = symbols["StableAdamW"](
        [{"params": trainable.parameters()}], lr=1e-3, betas=(0.9, 0.999),
        weight_decay=1e-4, amsgrad=True, eps=1e-10,
    )
    scheduler = symbols["WarmCosineScheduler"](
        optimizer, base_value=1e-3, final_value=1e-4,
        total_iters=epochs * len(loader), warmup_iters=100,
    )
    history: List[Dict[str, float]] = []
    start_epoch = 0
    if resume_state is not None:
        import random
        optimizer.load_state_dict(resume_state['optimizer'])
        scheduler.load_state_dict(resume_state['scheduler'])
        history = list(resume_state['history'])
        start_epoch = int(resume_state['epoch'])
        # Persistent workers consume a base seed only on their first iterator.
        # Prime the new worker pool before restoring the saved epoch-boundary RNG.
        if num_workers > 0:
            iter(loader)
        random.setstate(resume_state['python_rng'])
        np.random.set_state(resume_state['numpy_rng'])
        torch.set_rng_state(resume_state['torch_rng'])
        if resume_state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(resume_state['cuda_rng'], device)
    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        for images, _ in tqdm(loader, ncols=80):
            images = images.to(device, non_blocking=True)
            encoder_features, decoder_features, gather_loss = model(images)
            loss = symbols["global_cosine_hm_adaptive"](encoder_features, decoder_features, y=3)
            loss = loss + 0.2 * gather_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        mean_loss = float(np.mean(losses))
        print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, epochs, mean_loss), flush=True)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        if epoch_callback is not None:
            epoch_callback(epoch + 1, history, optimizer, scheduler)
    return history


def score_records(
    model: Any, records: Sequence[Record], transform: Any, gt_transform: Any,
    symbols: Dict[str, Any], device: Any, batch_size: int, num_workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset = EvaluationDataset(records, transform, gt_transform, OFFICIAL_CONFIG["crop_size"])
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        **data_loader_options(num_workers)
    )
    kernel = symbols["get_gaussian_kernel"](
        kernel_size=OFFICIAL_CONFIG["gaussian_kernel_size"], sigma=OFFICIAL_CONFIG["gaussian_sigma"]
    ).to(device)
    maps, masks, labels, paths, scores = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for images, gt, batch_labels, batch_paths in tqdm(loader, ncols=80):
            images = images.to(device, non_blocking=True)
            encoder_features, decoder_features = model(images)[:2]
            anomaly_map, _ = symbols["cal_anomaly_maps"](encoder_features, decoder_features, images.shape[-1])
            anomaly_map = functional.interpolate(
                anomaly_map, size=OFFICIAL_CONFIG["evaluation_map_size"], mode="bilinear", align_corners=False
            )
            anomaly_map = kernel(anomaly_map)
            gt = functional.interpolate(
                gt, size=OFFICIAL_CONFIG["evaluation_map_size"], mode="nearest"
            )
            gt = (gt > 0.5).to(torch.uint8)
            flat = anomaly_map.flatten(1)
            top_count = int(flat.shape[1] * 0.01)
            image_scores = torch.sort(flat, dim=1, descending=True)[0][:, :top_count].mean(dim=1)
            maps.append(anomaly_map[:, 0].cpu().numpy())
            masks.append(gt[:, 0].cpu().numpy())
            labels.append(batch_labels.flatten().cpu().numpy())
            scores.append(image_scores.cpu().numpy())
            paths.extend(list(batch_paths))
    return (
        np.concatenate(maps), np.concatenate(masks), np.concatenate(labels),
        paths, np.concatenate(scores),
    )


def evaluate_category(
    category: str, model: Any, records: Sequence[Record], transform: Any, gt_transform: Any,
    symbols: Dict[str, Any], device: Any, batch_size: int, num_workers: int,
    output_dir: Path, protocol: str, threshold: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    groups = sorted({(record.domain, record.shift) for record in records})
    for domain, shift in groups:
        group = [record for record in records if (record.domain, record.shift) == (domain, shift)]
        maps, masks, labels, paths, scores = score_records(
            model, group, transform, gt_transform, symbols, device, batch_size, num_workers
        )
        row: Dict[str, Any] = {
            "protocol": protocol,
            "category": category,
            "domain": domain,
            "shift": shift,
            "mask_capability": group[0].mask_capability,
            "images": len(group),
            "normal_images": int(np.sum(labels == 0)),
            "anomaly_images": int(np.sum(labels == 1)),
        }
        row.update(image_metrics(labels, scores, threshold if protocol == "scrs" else None))
        if category in PIXEL_CATEGORIES:
            values = symbols["ader_evaluator"](maps, scores, masks, labels)
            row.update({
                "pixel_AUROC_official_adeval": float(values[3]),
                "pixel_AUPR_official_adeval": float(values[4]),
                "pixel_AUPRO_official_adeval_nstrips200": float(values[6]),
            })
        for record, path, label, score in zip(group, paths, labels, scores):
            score_row = {
                "protocol": protocol, "category": category, "domain": domain, "shift": shift,
                "sample_id": record.sample_id, "path": path, "label": int(label), "image_score": float(score),
            }
            if protocol == "scrs":
                score_row["calibration_q95_threshold"] = float(threshold)
                score_row["predicted_anomaly"] = int(score > float(threshold))
            score_rows.append(score_row)
        group_rows.append(row)
        write_csv(output_dir / category / "{}_{}_image_scores.csv".format(domain, shift), score_rows[-len(group):])
    return group_rows, score_rows


def validate_backbone(official_root: Path, expected_hash: str) -> Dict[str, Any]:
    path = official_root / "backbones" / "weights" / BACKBONE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            "Missing pinned backbone {}. Download it explicitly before model execution.".format(path)
        )
    actual = file_hash(path)
    if not expected_hash:
        raise RuntimeError("--backbone-sha256 is mandatory for model execution")
    if expected_hash.lower() != EXPECTED_BACKBONE_SHA256:
        raise RuntimeError(
            "Supplied backbone approval hash differs from the locked official checkpoint hash"
        )
    if actual.lower() != expected_hash.lower():
        raise RuntimeError("Backbone SHA256 mismatch: expected {}, found {}".format(expected_hash, actual))
    return {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}


def run_experiment(args: argparse.Namespace) -> None:
    import torch

    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs/batch-size must be positive and num-workers must be non-negative")
    if args.protocol == "scrs" and abs(args.calibration_quantile - 0.95) > 1e-12:
        raise ValueError("This locked SCRS metric is q95; calibration-quantile must remain 0.95")
    validated_packages = validate_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("INP-Former model execution requires CUDA; local CPU execution is not supported")
    official = official_manifest(args.official_root)
    backbone = validate_backbone(args.official_root, args.backbone_sha256)
    train, raw_train, evaluation = robustad_records(args.data_root, args.categories)
    audit_rows, manifest_rows = audit_dataset(train, raw_train, evaluation)
    write_audit_outputs(args.output_dir, audit_rows, manifest_rows)
    failures = audit_failures(audit_rows)
    if failures:
        raise RuntimeError(
            "RobustAD audit failed for {} group(s); diagnostics were written to {}".format(
                len(failures), args.output_dir / "dataset_audit.json"
            )
        )
    if len(audit_rows) != 19 and not args.categories and not args.shifts:
        raise RuntimeError("Expected 19 formal evaluation groups, found {}".format(len(audit_rows)))

    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)
    symbols = import_official(args.official_root)
    device = torch.device("cuda:0")
    all_groups: List[Dict[str, Any]] = []
    all_scores: List[Dict[str, Any]] = []
    category_summaries: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    started = time.time()

    for category in train:
        symbols["setup_seed"](OFFICIAL_SEED)
        if args.protocol == "native":
            fit_records = list(train[category])
            calibration_records: List[Record] = []
        else:
            fit_records, calibration_records = scrs_partition(
                train[category], args.calibration_fraction, args.split_seed
            )
        for record in fit_records:
            split_rows.append({"category": category, "sample_id": record.sample_id, "role": "fit"})
        for record in calibration_records:
            split_rows.append({"category": category, "sample_id": record.sample_id, "role": "calibration"})

        transform, gt_transform = symbols["get_data_transforms"](
            OFFICIAL_CONFIG["input_size"], OFFICIAL_CONFIG["crop_size"]
        )
        model, trainable = build_model(symbols, args.official_root, device)
        torch.cuda.reset_peak_memory_stats(device)
        category_started = time.time()
        category_dir = args.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        history = train_model(
            model, trainable, fit_records, transform, symbols, device,
            args.epochs, args.batch_size, args.num_workers,
        )
        training_seconds = time.time() - category_started
        write_csv(category_dir / "training_history.csv", history)
        checkpoint = category_dir / "model.pth"
        torch.save(model.state_dict(), str(checkpoint))

        threshold = None
        if args.protocol == "scrs":
            _, _, cal_labels, cal_paths, cal_scores = score_records(
                model, calibration_records, transform, gt_transform, symbols, device,
                args.batch_size, args.num_workers,
            )
            if np.any(cal_labels != 0):
                raise RuntimeError("SCRS calibration set contains anomalous labels")
            threshold = float(np.quantile(cal_scores, args.calibration_quantile, interpolation="linear"))
            write_csv(
                category_dir / "calibration_scores.csv",
                [
                    {"category": category, "path": path, "label": int(label), "image_score": float(score),
                     "calibration_quantile": args.calibration_quantile, "threshold": threshold}
                    for path, label, score in zip(cal_paths, cal_labels, cal_scores)
                ],
            )

        selected = filter_evaluation(
            evaluation[category], args.shifts, args.max_eval_per_label_per_group
        )
        group_rows, score_rows = evaluate_category(
            category, model, selected, transform, gt_transform, symbols, device,
            args.batch_size, args.num_workers, args.output_dir, args.protocol, threshold,
        )
        all_groups.extend(group_rows)
        all_scores.extend(score_rows)
        category_summary = aggregate(group_rows, "within_category_group_macro", args.protocol)
        category_summary["category"] = category
        category_summary["fit_images"] = len(fit_records)
        category_summary["calibration_images"] = len(calibration_records)
        category_summary["checkpoint_sha256"] = file_hash(checkpoint)
        category_summary["training_seconds"] = training_seconds
        category_summary["category_total_seconds"] = time.time() - category_started
        category_summary["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        category_summary["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        if threshold is not None:
            category_summary["calibration_q95_threshold"] = threshold
        category_summaries.append(category_summary)
        write_json(category_dir / "summary.json", category_summary)

        del model, trainable
        gc.collect()
        torch.cuda.empty_cache()

    write_csv(args.output_dir / "protocol_split.csv", split_rows)
    write_csv(args.output_dir / "group_metrics.csv", all_groups)
    write_csv(args.output_dir / "image_scores.csv", all_scores)
    write_csv(args.output_dir / "category_metrics.csv", category_summaries)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": validated_packages,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    protocol = {
        "name": "INP-Former single-class full-shot external-dataset evaluation",
        "dataset": "RobustAD",
        "protocol": args.protocol,
        "official_seed": OFFICIAL_SEED,
        "official_config": OFFICIAL_CONFIG,
        "official_repository": official,
        "external_source_sha256": external_manifest(),
        "backbone": backbone,
        "categories": list(train),
        "selected_shifts": args.shifts,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "performance_plumbing": {
            "cuda_launch_blocking": False,
            "pin_memory": True,
            "persistent_workers": args.num_workers > 0,
            "prefetch_factor": 2 if args.num_workers > 0 else None,
            "non_blocking_host_to_device": True,
            "mixed_precision": False,
        },
        "scrs": None if args.protocol == "native" else {
            "split_seed": args.split_seed,
            "calibration_fraction": args.calibration_fraction,
            "calibration_quantile": args.calibration_quantile,
            "threshold_source": "held_out_source_normal_calibration_only",
        },
    }
    summary = {
        "protocol": protocol,
        "environment": environment,
        "runtime_seconds": time.time() - started,
        "evaluation_group_macro": aggregate(all_groups, "RobustAD_evaluation_group_macro", args.protocol),
        "source_group_macro": aggregate(
            [row for row in all_groups if row["domain"] == "source"],
            "RobustAD_source_group_macro", args.protocol,
        ),
        "target_group_macro": aggregate(
            [row for row in all_groups if row["domain"] == "target"],
            "RobustAD_target_group_macro", args.protocol,
        ),
        "category_macro": aggregate(category_summaries, "RobustAD_category_macro", args.protocol),
        "category_summaries": category_summaries,
    }
    write_json(args.output_dir / "protocol.json", protocol)
    write_json(args.output_dir / "environment.json", environment)
    write_json(args.output_dir / "summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "run"):
        current = subparsers.add_parser(command)
        current.add_argument("--data-root", type=Path, required=True)
        current.add_argument("--official-root", type=Path, required=True)
        current.add_argument("--output-dir", type=Path, required=True)
        current.add_argument("--categories", nargs="+", choices=tuple(ROBUSTAD_LAYOUT))
    run = subparsers.choices["run"]
    run.add_argument("--protocol", choices=("native", "scrs"), required=True)
    run.add_argument("--backbone-sha256", required=True)
    run.add_argument("--epochs", type=int, default=OFFICIAL_CONFIG["total_epochs"])
    run.add_argument("--batch-size", type=int, default=OFFICIAL_CONFIG["batch_size"])
    run.add_argument("--num-workers", type=int, default=OFFICIAL_CONFIG["num_workers"])
    run.add_argument("--shifts", nargs="+")
    run.add_argument("--max-eval-per-label-per-group", type=int)
    run.add_argument("--split-seed", type=int, default=42)
    run.add_argument("--calibration-fraction", type=float, default=0.20)
    run.add_argument("--calibration-quantile", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.official_root = args.official_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.command == "audit":
        official = official_manifest(args.official_root)
        train, raw_train, evaluation = robustad_records(args.data_root, args.categories)
        audit_rows, manifest_rows = audit_dataset(train, raw_train, evaluation)
        write_audit_outputs(args.output_dir, audit_rows, manifest_rows)
        write_json(
            args.output_dir / "audit_provenance.json",
            {"official_repository": official, "external_source_sha256": external_manifest()},
        )
        failures = audit_failures(audit_rows)
        if failures:
            raise RuntimeError(
                "RobustAD audit failed for {} group(s); diagnostics were written to {}".format(
                    len(failures), args.output_dir / "dataset_audit.json"
                )
            )
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
