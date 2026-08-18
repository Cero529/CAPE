import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from run_avdeepfake1m_feature_extraction import AVSR_REPO, FEATURE_DATASET, LOG_DIR, prepare


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


def find_invalid_features(split: str) -> list[Path]:
    root = FEATURE_DATASET / split
    return [p for p in root.rglob("features.npz") if not is_valid_feature_file(p)]


def build_index(split: str) -> dict[str, int]:
    os.chdir(AVSR_REPO)
    sys.path.insert(0, str(AVSR_REPO))
    from features_dimodif import get_video_paths, inpath2outdir

    index = {}
    paths = get_video_paths("avdeepfake1m", split, [])
    for i, video_path in enumerate(paths):
        feature_path = Path(inpath2outdir("avdeepfake1m", video_path, "mediapipe")) / "features.npz"
        index[str((AVSR_REPO / feature_path).resolve()).lower()] = i
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--timeout", type=int, default=900, help="Seconds allowed for one failed video.")
    args = parser.parse_args()

    prepare(AVSR_REPO)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    index = build_index(args.split)

    invalid = find_invalid_features(args.split)
    print(f"[fresh-repair] invalid before repair: {len(invalid)}")
    if not invalid:
        return

    env = os.environ.copy()
    env.setdefault("DIMODIF_FAST_FEATURES", "1")

    for n, feature_path in enumerate(invalid, 1):
        key = str(feature_path.resolve()).lower()
        if key not in index:
            print(f"[fresh-repair] cannot map feature to dataset index: {feature_path}")
            continue
        idx = index[key]
        logfile = LOG_DIR / f"repair_{args.split}_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        command = [
            sys.executable,
            "features_dimodif.py",
            "-d",
            "avdeepfake1m",
            "-l",
            "mediapipe",
            "-s",
            args.split,
            "-g",
            args.gpu,
            "-b",
            f"[{idx},{idx + 1}]",
        ]
        print(f"[fresh-repair] {n}/{len(invalid)} idx={idx} -> {feature_path}")
        with logfile.open("w", encoding="utf-8", errors="replace") as handle:
            try:
                result = subprocess.run(
                    command,
                    cwd=AVSR_REPO,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    timeout=args.timeout,
                )
                print(f"[fresh-repair] exit={result.returncode}; log={logfile}")
            except subprocess.TimeoutExpired:
                print(f"[fresh-repair] timeout after {args.timeout}s; log={logfile}")

    remaining = find_invalid_features(args.split)
    print(f"[fresh-repair] invalid after repair: {len(remaining)}")
    for path in remaining:
        print(path)
    raise SystemExit(1 if remaining else 0)


if __name__ == "__main__":
    main()
