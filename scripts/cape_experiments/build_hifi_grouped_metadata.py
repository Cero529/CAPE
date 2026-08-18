"""Build a leakage-safe HiFi-AVDF metadata split for CAPE experiments.

The source HiFi CSV alternates a real video and its generated counterpart.
Splitting individual CSV rows can therefore place the two members of a
matched pair in different partitions.  This utility:

1. groups rows by the underlying video identity;
2. keeps only complete, valid real/generated pairs;
3. assigns whole pairs to the manuscript's fixed 60/20/20 split; and
4. writes a JSON manifest with exact counts and file hashes.

The split is fixed across model seeds.  ``--split-seed`` controls only the
dataset partition and must not be varied in a matched multi-seed experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_from_project(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pair_key(row: Dict[str, str]) -> str:
    """Return the matched source identity shared by real/generated videos."""

    raw_path = (row.get("raw_video_path") or row.get("feature_path") or "").replace("\\", "/")
    raw_path = raw_path.replace("_generated.mp4", ".mp4")
    raw_path = raw_path.replace("_generated/mediapipe/features.npz", "/mediapipe/features.npz")
    return raw_path.lower()


def valid_feature(row: Dict[str, str]) -> Tuple[bool, str]:
    path = resolve_from_project(row["feature_path"])
    if not path.is_file():
        return False, "missing"
    try:
        with np.load(path, allow_pickle=True) as data:
            if "video_features" not in data.files or "audio_features" not in data.files:
                return False, "missing_arrays"
            video = data["video_features"]
            audio = data["audio_features"]
            if video.ndim != 2 or audio.ndim != 2:
                return False, f"invalid_rank:{video.ndim},{audio.ndim}"
            if video.shape[1] != 768 or audio.shape[1] != 768:
                return False, f"invalid_width:{video.shape},{audio.shape}"
    except Exception as exc:  # preserve the exact reason in the manifest
        return False, f"load_error:{type(exc).__name__}:{exc}"
    return True, "ok"


def allocate_split_counts(num_pairs: int, train_fraction: float, val_fraction: float) -> Tuple[int, int, int]:
    num_train = int(round(num_pairs * train_fraction))
    num_val = int(round(num_pairs * val_fraction))
    num_test = num_pairs - num_train - num_val
    if min(num_train, num_val, num_test) <= 0:
        raise ValueError(f"Split contains an empty partition for {num_pairs} pairs")
    return num_train, num_val, num_test


def read_hifi_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("dataset") == "hifi_avdf"]
    if not rows:
        raise RuntimeError(f"No HiFi-AVDF rows found in {path}")
    required = {"dataset", "feature_path", "is_fake", "task_id"}
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise RuntimeError(f"Source metadata is missing columns: {missing}")
    return rows, fieldnames


def main() -> None:
    parser = argparse.ArgumentParser(description="Create group-safe HiFi-AVDF metadata.")
    parser.add_argument("--source", default="data/cape_metadata.csv")
    parser.add_argument("--output", default="data/cape_metadata_hifi_grouped.csv")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must lie in (0, 1)")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must lie in (0, 1)")
    if args.train_fraction + args.val_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than 1")

    source = resolve_from_project(args.source)
    output = resolve_from_project(args.output)
    manifest_path = (
        resolve_from_project(args.manifest)
        if args.manifest
        else output.with_name(output.stem + "_manifest.json")
    )
    rows, fieldnames = read_hifi_rows(source)

    by_task_and_pair: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task_and_pair[row["task_id"]][pair_key(row)].append(row)

    output_rows: List[Dict[str, str]] = []
    manifest: Dict[str, object] = {
        "source_metadata": str(source),
        "source_sha256": sha256_file(source),
        "output_metadata": str(output),
        "split_seed": args.split_seed,
        "fractions": {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": 1.0 - args.train_fraction - args.val_fraction,
        },
        "pairing_rule": "strip _generated from raw_video_path and group the matched real/generated pair",
        "tasks": {},
        "dropped_pairs": [],
    }

    rng = random.Random(args.split_seed)
    for task_id in sorted(by_task_and_pair):
        valid_pairs: Dict[str, List[Dict[str, str]]] = {}
        for key, pair_rows in sorted(by_task_and_pair[task_id].items()):
            labels = sorted(int(row["is_fake"]) for row in pair_rows)
            reasons = []
            for row in pair_rows:
                ok, reason = valid_feature(row)
                if not ok:
                    reasons.append({"sample_id": row.get("sample_id", ""), "reason": reason})
            if len(pair_rows) != 2 or labels != [0, 1] or reasons:
                manifest["dropped_pairs"].append(
                    {
                        "task_id": task_id,
                        "pair_key": key,
                        "num_rows": len(pair_rows),
                        "labels": labels,
                        "feature_errors": reasons,
                    }
                )
                continue
            valid_pairs[key] = pair_rows

        keys = sorted(valid_pairs)
        rng.shuffle(keys)
        num_train, num_val, num_test = allocate_split_counts(
            len(keys), args.train_fraction, args.val_fraction
        )
        split_by_key = {}
        for key in keys[:num_train]:
            split_by_key[key] = "train"
        for key in keys[num_train : num_train + num_val]:
            split_by_key[key] = "val"
        for key in keys[num_train + num_val :]:
            split_by_key[key] = "test"

        split_pair_counts = {"train": num_train, "val": num_val, "test": num_test}
        split_row_counts = {name: count * 2 for name, count in split_pair_counts.items()}
        for key in sorted(valid_pairs):
            split = split_by_key[key]
            for row in valid_pairs[key]:
                copied = dict(row)
                copied["split"] = split
                output_rows.append(copied)

        manifest["tasks"][task_id] = {
            "valid_pairs": len(keys),
            "valid_rows": len(keys) * 2,
            "split_pairs": split_pair_counts,
            "split_rows": split_row_counts,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    manifest["total_valid_rows"] = len(output_rows)
    manifest["total_valid_pairs"] = len(output_rows) // 2
    manifest["output_sha256"] = sha256_file(output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"Wrote {output}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
