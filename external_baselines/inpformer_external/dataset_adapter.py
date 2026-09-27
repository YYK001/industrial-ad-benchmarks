"""Read-only RobustAD adapter for the pinned INP-Former evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROBUSTAD_LAYOUT = {
    "MetalParts": (
        "metal_parts_data_dir",
        {1: "lighting", 2: "position", 3: "rotation", 4: "scale", 5: "background_1", 6: "background_2"},
    ),
    "PCB": (
        "pcb_data_dir",
        {1: "lighting", 2: "white_balancing", 3: "rotation", 4: "position", 5: "shadow"},
    ),
    "PiledBags": (
        "piled_bags_data_dir",
        {1: "lighting", 2: "background_box_color", 3: "position_rotation", 4: "scale", 5: "shadow"},
    ),
}
PIXEL_CATEGORIES = {"MetalParts", "PCB"}


@dataclass(frozen=True)
class Record:
    sample_id: str
    path: str
    label: int
    mask_path: Optional[str]
    category: str
    domain: str
    shift: str
    role: str
    mask_capability: str = "not_evaluation"


def _read_split(category: str, split: Path, domain: str, shift: str, role: str) -> List[Record]:
    metadata = split / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    records: List[Record] = []
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "file_name" not in row or "label" not in row:
            raise ValueError("Missing file_name/label in {}:{}".format(metadata, line_number))
        relative = Path(str(row["file_name"]))
        image = split / relative
        if not image.is_file():
            raise FileNotFoundError(image)
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError("Non-binary label in {}:{}".format(metadata, line_number))
        mask_value = row.get("mask")
        mask_path = None
        if mask_value:
            raw_mask = Path(str(mask_value))
            candidates = (raw_mask, split / raw_mask, split / "masks" / raw_mask.name)
            mask_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
        sample_id = "{}/{}/{}".format(category, split.name, relative.as_posix())
        records.append(
            Record(
                sample_id=sample_id,
                path=str(image),
                label=label,
                mask_path=str(mask_path) if mask_path else None,
                category=category,
                domain=domain,
                shift=shift,
                role=role,
            )
        )
    if not records:
        raise RuntimeError("Empty RobustAD split: {}".format(split))
    return records


def robustad_records(
    root: Path, categories: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, List[Record]], Dict[str, List[Record]], Dict[str, List[Record]]]:
    """Return normal fit candidates, all raw training rows, and evaluation rows."""
    selected = tuple(categories or ROBUSTAD_LAYOUT)
    unknown = sorted(set(selected) - set(ROBUSTAD_LAYOUT))
    if unknown:
        raise ValueError("Unknown RobustAD categories: {}".format(unknown))
    fit_candidates: Dict[str, List[Record]] = {}
    raw_train: Dict[str, List[Record]] = {}
    evaluation: Dict[str, List[Record]] = {}
    for category in selected:
        prefix, shifts = ROBUSTAD_LAYOUT[category]
        category_root = root / category
        train_rows = _read_split(
            category, category_root / "{}_train".format(prefix), "source", "source_train", "train"
        )
        raw_train[category] = train_rows
        fit_candidates[category] = sorted(
            (record for record in train_rows if record.label == 0), key=lambda record: record.sample_id
        )
        if not fit_candidates[category]:
            raise RuntimeError("No source normal training images for {}".format(category))

        rows: List[Record] = []
        groups = [(0, "source", "source")]
        groups.extend((index, "target", shift) for index, shift in sorted(shifts.items()))
        for index, domain, shift in groups:
            rows.extend(
                _read_split(
                    category,
                    category_root / "{}_test{}".format(prefix, index),
                    domain,
                    shift,
                    "evaluation",
                )
            )

        resolved: List[Record] = []
        for domain, shift in sorted({(record.domain, record.shift) for record in rows}):
            group = [record for record in rows if (record.domain, record.shift) == (domain, shift)]
            if category in PIXEL_CATEGORIES:
                capability = "pixel"
            else:
                capability = "image"
            resolved.extend(replace(record, mask_capability=capability) for record in group)
        evaluation[category] = resolved
    return fit_candidates, raw_train, evaluation


def scrs_partition(
    records: Sequence[Record], calibration_fraction: float, split_seed: int
) -> Tuple[List[Record], List[Record]]:
    """Deterministic path-independent fit/calibration split made before training."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if len(records) < 2:
        raise RuntimeError("SCRS needs at least two normal training images")

    def rank(record: Record) -> str:
        payload = "{}\0{}\0{}".format(split_seed, record.category, record.sample_id)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    ordered = sorted(records, key=lambda record: (rank(record), record.sample_id))
    count = int(round(len(ordered) * calibration_fraction))
    count = min(max(count, 1), len(ordered) - 1)
    calibration_ids = {record.sample_id for record in ordered[:count]}
    fit = [record for record in records if record.sample_id not in calibration_ids]
    calibration = [record for record in records if record.sample_id in calibration_ids]
    if set(record.sample_id for record in fit) & set(record.sample_id for record in calibration):
        raise AssertionError("SCRS fit/calibration overlap")
    return fit, calibration


def filter_evaluation(
    records: Sequence[Record], shifts: Optional[Sequence[str]], max_per_label_per_group: Optional[int]
) -> List[Record]:
    selected_shifts = set(shifts or [])
    filtered = [record for record in records if not selected_shifts or record.shift in selected_shifts]
    if max_per_label_per_group is None:
        return filtered
    if max_per_label_per_group <= 0:
        raise ValueError("max_per_label_per_group must be positive")
    output: List[Record] = []
    keys = sorted({(record.domain, record.shift) for record in filtered})
    for domain, shift in keys:
        group = [record for record in filtered if (record.domain, record.shift) == (domain, shift)]
        for label in (0, 1):
            candidates = sorted((record for record in group if record.label == label), key=lambda record: record.sample_id)
            output.extend(candidates[:max_per_label_per_group])
    return output


def iter_all_records(
    raw_train: Dict[str, List[Record]], evaluation: Dict[str, List[Record]]
) -> Iterable[Record]:
    for category in raw_train:
        for record in raw_train[category]:
            yield record
        for record in evaluation[category]:
            yield record
