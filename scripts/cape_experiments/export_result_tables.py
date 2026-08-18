import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cape_reporting import build_rows, collect_result_dirs, write_latex_rows, write_markdown_table


def main():
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="Export Markdown and LaTeX rows from CAPE-compatible history.json files.")
    parser.add_argument("--result-root", default="results")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--title", default="CAPE Continual Learning Results")
    args = parser.parse_args()

    result_dirs = collect_result_dirs(args.result_root)
    rows = build_rows(result_dirs)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    write_markdown_table(rows, os.path.join(args.output_dir, "summary_table.md"), title=args.title)
    write_latex_rows(rows, os.path.join(args.output_dir, "summary_table_rows.tex"))
    print(f"[Tables] Found {len(rows)} result directories.")
    print(f"[Tables] Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
