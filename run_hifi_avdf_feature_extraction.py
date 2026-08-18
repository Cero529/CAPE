"""
Configure and launch HiFi-AVDF feature extraction for CAPE/DiMoDif.

Run from PowerShell in the deepfake3 environment:
    python run_hifi_avdf_feature_extraction.py --bounds "[0,10]"

The script expects:
    G:\\dimodif-main\\datasets\\HiFi-AVDF
    G:\\dimodif-main\\dimodif-main\\autoavsr
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
RAW_DATASET = WORKSPACE / "datasets" / "HiFi-AVDF"
FEATURE_DATASET = WORKSPACE / "datasets" / "HiFi-AVDF_emb"
DIMODIF_DATA_LINK = DIMODIF_REPO / "data" / "HiFi-AVDF_emb"
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


def check_paths() -> list[str]:
    missing = []
    required = [
        DIMODIF_REPO / "external" / "features.py",
        RAW_DATASET / "kling2.5",
        RAW_DATASET / "veo3.1",
        RAW_DATASET / "wan2.5",
        RAW_DATASET / "seedance1.0",
        AVSR_REPO / "pipelines",
        LOCAL_AUDIO_MODEL / "model.pth",
        LOCAL_AUDIO_MODEL / "model.json",
        LOCAL_VIDEO_MODEL / "model.pth",
        LOCAL_VIDEO_MODEL / "model.json",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path))
    return missing


def prepare() -> None:
    missing = check_paths()
    if missing:
        raise RuntimeError("Missing required paths:\n" + "\n".join(f"  - {item}" for item in missing))
    (AVSR_REPO / "datasets").mkdir(parents=True, exist_ok=True)
    FEATURE_DATASET.mkdir(parents=True, exist_ok=True)
    (DIMODIF_REPO / "data").mkdir(parents=True, exist_ok=True)
    ensure_junction(AVSR_REPO / "datasets" / "HiFi-AVDF", RAW_DATASET)
    ensure_junction(AVSR_REPO / "datasets" / "HiFi-AVDF_emb", FEATURE_DATASET)
    ensure_junction(DIMODIF_DATA_LINK, FEATURE_DATASET)
    shutil.copy2(DIMODIF_REPO / "external" / "features.py", AVSR_REPO / "features_dimodif.py")
    for name in ["kling2.5.csv", "veo3.1.csv", "wan2.5.csv", "seedance1.0.csv", "README.md"]:
        src = RAW_DATASET / name
        if src.exists():
            shutil.copy2(src, FEATURE_DATASET / name)
    log("HiFi-AVDF feature extraction setup complete.")


def launch(gpu: str, bounds: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"hifi_avdf_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    command = [
        sys.executable,
        "features_dimodif.py",
        "-d",
        "hifi_avdf",
        "-l",
        "mediapipe",
        "-s",
        "all",
        "-g",
        gpu,
    ]
    if bounds:
        command.extend(["-b", bounds])
    log(f"Starting HiFi-AVDF extraction. Log: {logfile}")
    with logfile.open("w", encoding="utf-8", errors="replace") as handle:
        env = os.environ.copy()
        env.setdefault("DIMODIF_FAST_FEATURES", "1")
        process = subprocess.Popen(
            command,
            cwd=str(AVSR_REPO),
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
        raise RuntimeError(f"HiFi-AVDF extraction failed with exit code {code}. See {logfile}")
    log("HiFi-AVDF extraction finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--bounds", default="", help='Optional range, e.g. "[0,10]" for a quick test.')
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare()
    if args.prepare_only:
        return
    launch(args.gpu, args.bounds)


if __name__ == "__main__":
    main()
