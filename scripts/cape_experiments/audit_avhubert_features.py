"""Audit the frozen-feature contract used by the manuscript profile."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cape_datasets import explicit_valid_length


REQUIRED_KEYS = ("video_features", "audio_features", "backbone_features")


def _resolve_path(value, root_dir):
    path = Path(value)
    return path if path.is_absolute() else root_dir / path


def _shape_key(array):
    return "x".join(str(value) for value in array.shape)


def audit_file(path, expected_length, expected_dim, allow_unpadded, check_zero_padding):
    problems = []
    details = {}
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            return [f"missing NPZ keys: {missing}"], details

        video = np.asarray(data["video_features"])
        audio = np.asarray(data["audio_features"])
        backbone = np.asarray(data["backbone_features"])
        valid_length = explicit_valid_length(data)
        details.update(
            {
                "video_shape": _shape_key(video),
                "audio_shape": _shape_key(audio),
                "backbone_shape": _shape_key(backbone),
                "valid_length": valid_length,
            }
        )

        for name, value in (("video_features", video), ("audio_features", audio)):
            if value.ndim != 2:
                problems.append(f"{name} must be [T, D], got {value.shape}")
                continue
            if value.shape[1] != expected_dim:
                problems.append(f"{name} dimension must be {expected_dim}, got {value.shape[1]}")
            if allow_unpadded:
                if not 1 <= value.shape[0] <= expected_length:
                    problems.append(
                        f"{name} length must lie in [1, {expected_length}], got {value.shape[0]}"
                    )
            elif value.shape[0] != expected_length:
                problems.append(f"{name} length must be {expected_length}, got {value.shape[0]}")
            if not np.issubdtype(value.dtype, np.floating):
                problems.append(f"{name} must use a floating dtype, got {value.dtype}")
            elif not np.isfinite(value).all():
                problems.append(f"{name} contains NaN or Inf")

        if backbone.ndim == 1:
            if backbone.shape[0] != expected_dim:
                problems.append(
                    f"pooled backbone_features dimension must be {expected_dim}, got {backbone.shape[0]}"
                )
            details["backbone_kind"] = "pooled"
        elif backbone.ndim == 2:
            details["backbone_kind"] = "sequence"
            if backbone.shape[1] != expected_dim:
                problems.append(
                    f"backbone_features dimension must be {expected_dim}, got {backbone.shape[1]}"
                )
            if allow_unpadded:
                if not 1 <= backbone.shape[0] <= expected_length:
                    problems.append(
                        f"backbone sequence length must lie in [1, {expected_length}], got {backbone.shape[0]}"
                    )
            elif backbone.shape[0] != expected_length:
                problems.append(
                    f"backbone sequence length must be {expected_length}, got {backbone.shape[0]}"
                )
        else:
            problems.append(f"backbone_features must be [D] or [T, D], got {backbone.shape}")
        if not np.issubdtype(backbone.dtype, np.floating):
            problems.append(f"backbone_features must use a floating dtype, got {backbone.dtype}")
        elif not np.isfinite(backbone).all():
            problems.append("backbone_features contains NaN or Inf")

        if valid_length is None:
            problems.append("missing scalar valid_length or contiguous pair_valid_mask")
        elif video.ndim == 2 and audio.ndim == 2:
            available = min(video.shape[0], audio.shape[0], expected_length)
            if not 1 <= valid_length <= available:
                problems.append(f"valid_length must lie in [1, {available}], got {valid_length}")
            elif backbone.ndim == 2 and valid_length > backbone.shape[0]:
                problems.append(
                    f"valid_length={valid_length} exceeds backbone sequence length {backbone.shape[0]}"
                )
            elif check_zero_padding and valid_length < min(video.shape[0], audio.shape[0]):
                if not np.allclose(video[valid_length:], 0.0, atol=1e-6):
                    problems.append("video padding after valid_length is not zero")
                if not np.allclose(audio[valid_length:], 0.0, atol=1e-6):
                    problems.append("audio padding after valid_length is not zero")

    return problems, details


def main():
    parser = argparse.ArgumentParser(description="Audit NPZ files for the paper AV-HuBERT feature contract.")
    parser.add_argument("--metadata", required=True, help="Unified CAPE metadata CSV.")
    parser.add_argument(
        "--root-dir",
        default=str(PROJECT_ROOT),
        help="Base directory for relative feature_path values (default: project root).",
    )
    parser.add_argument("--expected-length", type=int, default=100)
    parser.add_argument("--expected-dim", type=int, default=768)
    parser.add_argument("--max-samples", type=int, default=0, help="0 audits every unique feature file.")
    parser.add_argument("--allow-unpadded", action="store_true")
    parser.add_argument("--check-zero-padding", action="store_true")
    parser.add_argument("--max-error-examples", type=int, default=20)
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    metadata = Path(args.metadata).resolve()
    root_dir = Path(args.root_dir).resolve()
    seen_paths = set()
    counters = Counter()
    shapes = {
        "video": Counter(),
        "audio": Counter(),
        "backbone": Counter(),
        "backbone_kind": Counter(),
    }
    errors = []

    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "feature_path" not in (reader.fieldnames or []):
            raise RuntimeError("metadata CSV is missing the feature_path column")
        for row_number, row in enumerate(reader, start=2):
            if args.max_samples > 0 and counters["checked_files"] >= args.max_samples:
                break
            counters["metadata_rows_scanned"] += 1
            path = _resolve_path(str(row.get("feature_path", "")), root_dir).resolve()
            normalized = os.path.normcase(str(path))
            if normalized in seen_paths:
                counters["duplicate_paths"] += 1
                continue
            seen_paths.add(normalized)
            counters["checked_files"] += 1
            if not path.is_file():
                counters["invalid_files"] += 1
                if len(errors) < args.max_error_examples:
                    errors.append({"row": row_number, "path": str(path), "problems": ["file not found"]})
                continue
            try:
                problems, details = audit_file(
                    path,
                    expected_length=args.expected_length,
                    expected_dim=args.expected_dim,
                    allow_unpadded=args.allow_unpadded,
                    check_zero_padding=args.check_zero_padding,
                )
            except Exception as exc:
                problems, details = [f"cannot read NPZ: {type(exc).__name__}: {exc}"], {}

            if details:
                shapes["video"][details["video_shape"]] += 1
                shapes["audio"][details["audio_shape"]] += 1
                shapes["backbone"][details["backbone_shape"]] += 1
                shapes["backbone_kind"][details.get("backbone_kind", "invalid")] += 1
                if details.get("valid_length") is not None:
                    counters["explicit_valid_length"] += 1
            if problems:
                counters["invalid_files"] += 1
                if len(errors) < args.max_error_examples:
                    errors.append({"row": row_number, "path": str(path), "problems": problems})
            else:
                counters["valid_files"] += 1

    report = {
        "metadata": str(metadata),
        "root_dir": str(root_dir),
        "profile": {
            "required_keys": list(REQUIRED_KEYS),
            "expected_length": args.expected_length,
            "expected_dim": args.expected_dim,
            "allow_unpadded": args.allow_unpadded,
            "check_zero_padding": args.check_zero_padding,
        },
        "counts": dict(counters),
        "shapes": {name: dict(values) for name, values in shapes.items()},
        "errors": errors,
        "passed": counters["checked_files"] > 0 and counters["invalid_files"] == 0,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
