"""
CLI:
  retention run --out reports/run1 [--save-rates 0.1 0.2 0.3 0.5 1.0] [--primary 0.3] [--control 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .pipeline import run_pipeline
from .policy import PolicyConfig


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="retention", description="Budget-constrained retention targeting pipeline.")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--data-dir", default=None)
    r.add_argument("--save-rates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 1.0])
    r.add_argument("--primary", type=float, default=0.3, help="save rate used for the primary offer list")
    r.add_argument("--control", type=int, default=10, help="held-out customers in the learn-while-earning list")
    r.add_argument("--support-relevance", type=float, default=0.5)
    args = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    cfg = PolicyConfig(save_rates=args.save_rates, primary_save_rate=args.primary, control_size=args.control)
    cfg.relevance["customer_support/negative"] = args.support_relevance
    ctx = run_pipeline(Path(args.out), cfg, Path(args.data_dir) if args.data_dir else None)
    print((Path(args.out) / "MEMO.md").read_text(encoding="utf-8"))
    return 0 if ctx["critic"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
