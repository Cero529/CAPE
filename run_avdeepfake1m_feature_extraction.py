"""
Configure and launch AV-Deepfake1M feature extraction for DiMoDif.

Run from PowerShell:
    python G:\\dimodif-main\\dimodif-main\\run_avdeepfake1m_feature_extraction.py

This script expects the Auto-AVSR repository:
    G:\\dimodif-main\\dimodif-main\\autoavsr

and the two Auto-AVSR checkpoints:
    G:\\dimodif-main\\dimodif-main\\autoavsr\\data\\LRS3_A_WER1.0\\model.pth
    G:\\dimodif-main\\dimodif-main\\autoavsr\\data\\LRS3_A_WER1.0\\model.json
    G:\\dimodif-main\\dimodif-main\\autoavsr\\data\\LRS3_V_WER19.1\\model.pth
    G:\\dimodif-main\\dimodif-main\\autoavsr\\data\\LRS3_V_WER19.1\\model.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DIMODIF_REPO = Path(__file__).resolve().parent
WORKSPACE = DIMODIF_REPO.parent
RAW_DATASET = WORKSPACE / "datasets" / "AV-Deepfake1M"
FEATURE_DATASET = WORKSPACE / "datasets" / "AV-Deepfake1M_emb"
DIMODIF_DATA_LINK = DIMODIF_REPO / "data" / "AV-Deepfake1M_emb"
AVSR_REPO = DIMODIF_REPO / "autoavsr"
LOG_DIR = WORKSPACE / "logs"
LOCAL_AUDIO_MODEL = AVSR_REPO / "data" / "LRS3_A_WER1.0"
LOCAL_VIDEO_MODEL = AVSR_REPO / "data" / "LRS3_V_WER19.1"


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def run(command: list[str], cwd: Path | None = None) -> None:
    log("RUN " + " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_junction(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.exists():
        if link.resolve() == target:
            return
        raise RuntimeError(f"{link} already exists and does not point to {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        run(["cmd", "/c", "mklink", "/J", str(link), str(target)])
    else:
        link.symlink_to(target, target_is_directory=True)


def copy_required_files(avsr_repo: Path) -> None:
    source = DIMODIF_REPO / "external" / "features.py"
    target = avsr_repo / "features_dimodif.py"
    shutil.copy2(source, target)
    log(f"Copied {source} -> {target}")


def link_local_models_into_avsr(avsr_repo: Path) -> None:
    data_dir = avsr_repo / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    log(f"Using LRS3 models from {data_dir}")


def copy_metadata() -> None:
    FEATURE_DATASET.mkdir(parents=True, exist_ok=True)
    for name in [
        "train_metadata.json",
        "val_metadata.json",
        "test_files.txt",
        "README.md",
        "LICENSE",
        "TERMS_AND_CONDITIONS.md",
    ]:
        src = RAW_DATASET / name
        if src.exists():
            shutil.copy2(src, FEATURE_DATASET / name)
    log(f"Metadata copied into {FEATURE_DATASET}")


def check_paths(avsr_repo: Path) -> list[str]:
    missing: list[str] = []
    required = [
        DIMODIF_REPO / "external" / "features.py",
        RAW_DATASET / "train",
        RAW_DATASET / "val",
        RAW_DATASET / "test",
        RAW_DATASET / "train_metadata.json",
        RAW_DATASET / "val_metadata.json",
        avsr_repo / "pipelines",
        LOCAL_AUDIO_MODEL / "model.pth",
        LOCAL_AUDIO_MODEL / "model.json",
        LOCAL_VIDEO_MODEL / "model.pth",
        LOCAL_VIDEO_MODEL / "model.json",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path))
    return missing


def prepare(avsr_repo: Path) -> None:
    if not avsr_repo.exists():
        raise RuntimeError(
            f"Auto-AVSR repo not found: {avsr_repo}\n"
            "Clone it first:\n"
            "  cd /d G:\\dimodif-main\n"
            "  git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages\n"
        )

    missing = check_paths(avsr_repo)
    if missing:
        raise RuntimeError(
            "Cannot start extraction because these required files/folders are missing:\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\n\nAuto-AVSR code must contain a pipelines/ folder. The LRS3 model folders should stay under:\n"
            + f"  {avsr_repo / 'data'}\n"
        )

    (avsr_repo / "datasets").mkdir(parents=True, exist_ok=True)
    FEATURE_DATASET.mkdir(parents=True, exist_ok=True)
    (DIMODIF_REPO / "data").mkdir(parents=True, exist_ok=True)

    ensure_junction(avsr_repo / "datasets" / "AV-Deepfake1M", RAW_DATASET)
    ensure_junction(avsr_repo / "datasets" / "AV-Deepfake1M_emb", FEATURE_DATASET)
    ensure_junction(DIMODIF_DATA_LINK, FEATURE_DATASET)
    link_local_models_into_avsr(avsr_repo)
    copy_required_files(avsr_repo)
    copy_metadata()


def launch_split(avsr_repo: Path, split: str, gpu: str, bounds: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"avdeepfake1m_extract_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    command = [
        sys.executable,
        "features_dimodif.py",
        "-d",
        "avdeepfake1m",
        "-l",
        "mediapipe",
        "-s",
        split,
        "-g",
        gpu,
    ]
    if bounds:
        command.extend(["-b", bounds])
    log(f"Starting split={split}. Log: {logfile}")
    with logfile.open("w", encoding="utf-8", errors="replace") as handle:
        env = os.environ.copy()
        env.setdefault("DIMODIF_FAST_FEATURES", "1")
        process = subprocess.Popen(
            command,
            cwd=str(avsr_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Extraction failed for split={split} with exit code {code}. See {logfile}")
    log(f"Finished split={split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avsr-repo",
        default=str(AVSR_REPO),
        help="Path to Visual_Speech_Recognition_for_Multiple_Languages.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA GPU index, e.g. 0.")
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits to convert. Default converts everything: train,val,test.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create links and check dependencies; do not start conversion.",
    )
    parser.add_argument(
        "--bounds",
        default="",
        help="Optional range passed to DiMoDif features.py, e.g. \"[0,100]\" for a quick speed test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    avsr_repo = Path(args.avsr_repo)
    prepare(avsr_repo)
    if args.prepare_only:
        log("Prepare-only complete. No extraction started.")
        return

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    valid = {"train", "val", "test", "all"}
    bad = [s for s in splits if s not in valid]
    if bad:
        raise ValueError(f"Invalid split(s): {bad}. Valid: {sorted(valid)}")
    for split in splits:
        launch_split(avsr_repo, split, args.gpu, args.bounds)
    copy_metadata()
    log("All requested extraction jobs finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log("ERROR: " + str(exc))
        if sys.stdin.isatty():
            input("Press Enter to exit...")
        raise
