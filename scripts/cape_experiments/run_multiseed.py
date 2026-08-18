import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Run a CAPE-compatible script over matched random seeds.")
    parser.add_argument("script", help="Training script path relative to the project root.")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-mode", choices=["dir", "root"], default="dir")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()

    script = args.script if os.path.isabs(args.script) else os.path.join(PROJECT_ROOT, args.script)
    extra = list(extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    output_flag = "--output-dir" if args.output_mode == "dir" else "--output-root"
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        output_path = os.path.join(args.output_root, f"seed_{seed}")
        command = [
            args.python,
            script,
            *extra,
            "--seed",
            str(seed),
            output_flag,
            output_path,
        ]
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
