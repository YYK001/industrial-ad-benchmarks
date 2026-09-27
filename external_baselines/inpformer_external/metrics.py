"""Metric and provenance helpers for the INP-Former RobustAD evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


NATIVE_METRICS = (
    "image_AUROC",
    "image_AUPR",
    "pixel_AUROC_official_adeval",
    "pixel_AUPR_official_adeval",
    "pixel_AUPRO_official_adeval_nstrips200",
)
SCRS_METRICS = NATIVE_METRICS + ("normal_image_FP_calibration_q95",)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("Refusing to write empty CSV: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: Optional[float] = None) -> Dict[str, float]:
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Image AUROC/AUPR requires both normal and anomaly labels")
    result = {
        "image_AUROC": float(roc_auc_score(labels, scores)),
        "image_AUPR": float(average_precision_score(labels, scores)),
    }
    if threshold is not None:
        normal = scores[labels == 0]
        result["normal_image_FP_calibration_q95"] = float(np.mean(normal > threshold))
    return result


def aggregate(rows: Sequence[Dict[str, Any]], aggregation: str, protocol: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "aggregation": aggregation,
        "protocol": protocol,
        "pixel_scope": "MetalParts_and_PCB_groups_only",
    }
    keys = SCRS_METRICS if protocol == "scrs" else NATIVE_METRICS
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and row[key] is not None and np.isfinite(float(row[key]))]
        if values:
            result[key] = float(np.mean(values))
            result[key + "_count"] = len(values)
    return result
