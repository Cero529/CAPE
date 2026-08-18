import argparse
import csv
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_datasets import infer_pattern_id


def add_avdeepfake1m_rows(rows, root, split, detector_name="mediapipe"):
    metadata_path = os.path.join(root, "data", "AV-Deepfake1M_emb", f"{split}_metadata.json")
    feature_root = os.path.join(root, "data", "AV-Deepfake1M_emb", split)
    if not os.path.exists(metadata_path):
        return
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    label_map = {
        "real": (0, 0),
        "visual_modified": (1, 0),
        "audio_modified": (0, 1),
        "both_modified": (1, 1),
    }
    for item in metadata:
        video_target, audio_target = label_map.get(item.get("modify_type", "real"), (0, 0))
        rel = item["file"].replace(".mp4", "")
        feature_path = os.path.join(feature_root, rel, detector_name, "features.npz")
        pattern_id = infer_pattern_id(video_target, audio_target, fake_periods=None)
        rows.append(
            {
                "sample_id": f"avdeepfake1m:{split}:{item['file']}",
                "dataset": "avdeepfake1m",
                "feature_path": feature_path,
                "split": "val" if split == "dev" else split,
                "video_target": video_target,
                "audio_target": audio_target,
                "is_fake": int(video_target or audio_target),
                "pattern_id": pattern_id,
                "task_id": pattern_id if pattern_id != 0 else "real",
                "generator": "unknown",
                "fake_periods": item.get("fake_segments", ""),
                "video_available": 1,
                "audio_available": 1,
                "audio_visual_available": 1,
                "temporal_available": int("fake_segments" in item and item.get("fake_segments") is not None),
                "unknown_flag": 0,
                "unknown_available": 1,
            }
        )


def _hifi_split(index, n_rows):
    frac = index / max(1, n_rows)
    if frac < 0.7:
        return "train"
    if frac < 0.85:
        return "val"
    return "test"


def add_hifi_rows(rows, hifi_dir, detector_name="mediapipe"):
    if not os.path.isdir(hifi_dir):
        return
    for name in os.listdir(hifi_dir):
        if not name.endswith(".csv"):
            continue
        generator = name.replace(".csv", "")
        with open(os.path.join(hifi_dir, name), newline="") as f:
            table = list(csv.DictReader(f))
        for idx, item in enumerate(table):
            video_path = item.get("video_path", "")
            feature_path = os.path.join(
                "data",
                "HiFi-AVDF_emb",
                os.path.splitext(video_path)[0],
                detector_name,
                "features.npz",
            )
            audio_target = int(item.get("audio_label", item.get("audio_target", 1)))
            video_target = int(item.get("visual_label", item.get("video_target", 1)))
            is_fake = int(item.get("overall_label", item.get("is_fake", int(audio_target or video_target))))
            rows.append(
                {
                    "sample_id": f"hifi:{generator}:{idx}",
                    "dataset": "hifi_avdf",
                    "feature_path": feature_path,
                    "raw_video_path": os.path.join("..", "datasets", "HiFi-AVDF", video_path),
                    "split": item.get("split", _hifi_split(idx, len(table))),
                    "video_target": video_target,
                    "audio_target": audio_target,
                    "is_fake": is_fake,
                    "pattern_id": 5,
                    "task_id": f"generator:{generator}",
                    "generator": generator,
                    "fake_periods": item.get("fake_periods", ""),
                    "video_available": 1,
                    "audio_available": 1,
                    "audio_visual_available": 1,
                    "temporal_available": int("fake_periods" in item and item.get("fake_periods") is not None),
                    "unknown_flag": 0,
                    "unknown_available": 1,
                }
            )


def is_valid_feature_file(project_root, feature_path):
    path = feature_path
    if not os.path.isabs(path):
        path = os.path.join(project_root, path)
    if not os.path.exists(path):
        return False
    try:
        data = np.load(path, allow_pickle=True)
        return (
            "video_features" in data.files
            and "audio_features" in data.files
            and data["video_features"].ndim == 2
            and data["audio_features"].ndim == 2
            and data["video_features"].shape[1] == 768
            and data["audio_features"].shape[1] == 768
        )
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Build CAPE unified metadata CSV.")
    parser.add_argument("--root", default=".", help="Project root, e.g. G:/dimodif-main/dimodif-main")
    parser.add_argument("--hifi-dir", default="../datasets/HiFi-AVDF")
    parser.add_argument("--output", default="data/cape_metadata.csv")
    parser.add_argument("--detector-name", default="mediapipe")
    parser.add_argument(
        "--skip-invalid-hifi",
        action="store_true",
        help="Drop HiFi-AVDF rows whose extracted features are missing or invalid.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Drop any row whose extracted features are missing or invalid.",
    )
    args = parser.parse_args()

    rows = []
    for split in ("train", "val"):
        add_avdeepfake1m_rows(rows, args.root, split, detector_name=args.detector_name)
    add_hifi_rows(rows, args.hifi_dir, detector_name=args.detector_name)

    if args.skip_invalid or args.skip_invalid_hifi:
        before = len(rows)
        if args.skip_invalid:
            rows = [row for row in rows if is_valid_feature_file(args.root, row["feature_path"])]
            print(f"Skipped {before - len(rows)} invalid rows.")
        else:
            rows = [
                row
                for row in rows
                if row["dataset"] != "hifi_avdf" or is_valid_feature_file(args.root, row["feature_path"])
            ]
            print(f"Skipped {before - len(rows)} invalid HiFi-AVDF rows.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    counts = {}
    for row in rows:
        key = (row["dataset"], row["task_id"])
        counts[key] = counts.get(key, 0) + 1
    for key, value in sorted(counts.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        print(f"{key[0]} {key[1]}: {value}")


if __name__ == "__main__":
    main()
