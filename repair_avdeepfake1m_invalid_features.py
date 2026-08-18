import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch

from run_avdeepfake1m_feature_extraction import AVSR_REPO, FEATURE_DATASET, prepare


def is_valid_feature_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 2048:
        return False
    try:
        data = np.load(path, allow_pickle=True)
        return (
            "video_features" in data.files
            and "audio_features" in data.files
            and data["video_features"].shape != ()
            and data["audio_features"].shape != ()
        )
    except Exception:
        return False


def feature_to_video_path(feature_path: Path, split: str) -> str:
    split_root = FEATURE_DATASET / split
    rel = feature_path.relative_to(split_root)
    parts = list(rel.parts)
    # id / youtube_id / clip_id / video_stem / mediapipe / features.npz
    video_stem = parts[-3]
    video_rel = Path(*parts[:-3]) / f"{video_stem}.mp4"
    return str(Path("datasets") / "AV-Deepfake1M" / split / video_rel)


def find_invalid_features(split: str) -> list[Path]:
    root = FEATURE_DATASET / split
    if not root.exists():
        return []
    return [p for p in root.rglob("features.npz") if not is_valid_feature_file(p)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-rounds", type=int, default=3)
    args = parser.parse_args()

    os.environ.setdefault("DIMODIF_FAST_FEATURES", "1")
    prepare(AVSR_REPO)
    os.chdir(AVSR_REPO)
    sys.path.insert(0, str(AVSR_REPO))

    from features_dimodif import get_pipeline_obj, store_videos_embeddings

    for round_idx in range(1, args.max_rounds + 1):
        invalid = find_invalid_features(args.split)
        print(f"[repair] round={round_idx} invalid={len(invalid)}")
        if not invalid:
            print("[repair] all feature files are valid.")
            return

        video_paths = [feature_to_video_path(path, args.split) for path in invalid]
        pipeline_audio = get_pipeline_obj("audio", args.gpu)
        pipeline_video = get_pipeline_obj("video", args.gpu)
        store_videos_embeddings(
            "avdeepfake1m",
            video_paths,
            pipeline_audio,
            pipeline_video,
            detector_name="mediapipe",
            detector=None,
        )
        del pipeline_audio, pipeline_video
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    invalid = find_invalid_features(args.split)
    if invalid:
        print("[repair] still invalid after retries:")
        for path in invalid:
            print(path)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
