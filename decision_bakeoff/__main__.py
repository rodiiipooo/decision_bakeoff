"""CLI entry: python -m decision_bakeoff run --data data/samples --arch A0,A1,..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from decision_bakeoff import ARCHITECTURES, __version__
from decision_bakeoff.harness import run_bakeoff


def _parse_arch(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return list(ARCHITECTURES)
    unknown = [p for p in parts if p not in ARCHITECTURES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown arch ids: {unknown}; choose from {list(ARCHITECTURES)}"
        )
    return parts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m decision_bakeoff",
        description="Offline/shadow decision architecture bakeoff (A0–A5).",
    )
    p.add_argument("--version", action="version", version=f"decision_bakeoff {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run bakeoff over JSONL samples")
    run_p.add_argument(
        "--data",
        type=Path,
        default=Path("data/samples"),
        help="Directory of *.jsonl example files",
    )
    run_p.add_argument(
        "--arch",
        type=_parse_arch,
        default=list(ARCHITECTURES),
        help="Comma-separated architecture ids (default: all A0–A5)",
    )
    run_p.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        help="Results root directory",
    )
    run_p.add_argument(
        "--splits",
        default="shadow,holdout",
        help="Comma-separated eval splits (default: shadow,holdout)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        out = run_bakeoff(
            data_dir=args.data,
            arch_ids=args.arch,
            results_root=args.results,
            eval_splits=splits,
        )
        print(f"Bakeoff complete: {out}")
        print(f"  report:  {out / 'report.md'}")
        print(f"  metrics: {out / 'metrics.json'}")
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
