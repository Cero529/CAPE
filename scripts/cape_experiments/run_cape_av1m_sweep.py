import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "cape_pattern_incremental_av1M.py")


STARTER_CONFIGS = [
    {"name": "e3_len512_bs32_lr1e-4_rep1024", "epochs": 3, "max_length": 512, "batch_size": 32, "lr": "1e-4", "replay_capacity": 1024},
    {"name": "e5_len512_bs32_lr1e-4_rep1024", "epochs": 5, "max_length": 512, "batch_size": 32, "lr": "1e-4", "replay_capacity": 1024},
    {"name": "e5_len512_bs32_lr3e-4_rep1024", "epochs": 5, "max_length": 512, "batch_size": 32, "lr": "3e-4", "replay_capacity": 1024},
    {"name": "e5_len600_bs32_lr1e-4_rep1024", "epochs": 5, "max_length": 600, "batch_size": 32, "lr": "1e-4", "replay_capacity": 1024},
    {"name": "e5_len512_bs32_lr1e-4_rep2048", "epochs": 5, "max_length": 512, "batch_size": 32, "lr": "1e-4", "replay_capacity": 2048},
    {"name": "e8_len512_bs32_lr1e-4_rep2048", "epochs": 8, "max_length": 512, "batch_size": 32, "lr": "1e-4", "replay_capacity": 2048},
]


FULL_EXTRA_CONFIGS = [
    {"name": "e5_len384_bs32_lr1e-4_rep1024", "epochs": 5, "max_length": 384, "batch_size": 32, "lr": "1e-4", "replay_capacity": 1024},
    {"name": "e5_len512_bs64_lr1e-4_rep1024", "epochs": 5, "max_length": 512, "batch_size": 64, "lr": "1e-4", "replay_capacity": 1024},
    {"name": "e8_len512_bs32_lr5e-5_rep2048", "epochs": 8, "max_length": 512, "batch_size": 32, "lr": "5e-5", "replay_capacity": 2048},
    {"name": "e10_len512_bs32_lr1e-4_rep2048", "epochs": 10, "max_length": 512, "batch_size": 32, "lr": "1e-4", "replay_capacity": 2048},
]


def build_command(python_exe, config, output_root, args):
    command = [
        python_exe,
        TRAIN_SCRIPT,
        "--epochs",
        str(config["epochs"]),
        "--max-length",
        str(config["max_length"]),
        "--batch-size",
        str(config["batch_size"]),
        "--lr",
        str(config["lr"]),
        "--replay-capacity",
        str(config["replay_capacity"]),
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        os.path.join(output_root, config["name"]),
    ]
    if args.amp:
        command.append("--amp")
    if args.no_pin_memory:
        command.append("--no-pin-memory")
    if args.no_persistent_workers:
        command.append("--no-persistent-workers")
    return command


def main():
    parser = argparse.ArgumentParser(description="Run a preset CAPE AV1M hyperparameter sweep.")
    parser.add_argument("--preset", choices=["starter", "full"], default="starter")
    parser.add_argument("--output-root", default="results/cape_av1m_sweep")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-at", type=int, default=0, help="Skip configs before this zero-based index.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    configs = list(STARTER_CONFIGS)
    if args.preset == "full":
        configs.extend(FULL_EXTRA_CONFIGS)

    for index, config in enumerate(configs):
        if index < args.start_at:
            continue
        command = build_command(args.python, config, args.output_root, args)
        print(f"\n[Sweep] {index}/{len(configs) - 1}: {config['name']}")
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
