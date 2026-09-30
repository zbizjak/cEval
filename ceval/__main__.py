"""cEval CLI: two centerline graphs in, one JSON report out.

    python3 -m ceval pred.vtp gt.vtp --d-mm 0.7
    python3 -m ceval pred.vtp gt.vtp --dataset topcow_mr -o report.json
"""
import argparse
import json
import sys
from pathlib import Path

from ceval.io import read_vtp_polylines
from ceval.report import SpacingMismatch, build

# Matching tolerance (dataset mean voxel diagonal, mm) used in the paper.
D_MM = {"topcow_mr": 0.7, "topcow_ct": 0.9, "aortaseg24": 1.6, "imagecas": 0.7}


def main():
    ap = argparse.ArgumentParser(prog="ceval", description=__doc__)
    ap.add_argument("prediction", help="predicted graph (.vtp), mm coordinates")
    ap.add_argument("reference", help="reference graph (.vtp), mm coordinates")
    ap.add_argument("--d-mm", type=float,
                    help="matching tolerance in mm (mean voxel diagonal)")
    ap.add_argument("--dataset", help="look D up by dataset name instead")
    ap.add_argument("-o", "--output", help="write here (default: stdout)")
    ap.add_argument("--force", action="store_true",
                    help="evaluate despite a failed spacing check; the matching "
                         "metrics will not be interpretable")
    args = ap.parse_args()

    if args.d_mm is None:
        if not args.dataset:
            ap.error("one of --d-mm or --dataset is required")
        if args.dataset not in D_MM:
            ap.error(f"unknown dataset {args.dataset!r}; known: {sorted(D_MM)}")
        args.d_mm = D_MM[args.dataset]

    pred = read_vtp_polylines(args.prediction)
    gt = read_vtp_polylines(args.reference)
    try:
        report = build(*pred, *gt, args.d_mm,
                       pred_name=Path(args.prediction).name,
                       gt_name=Path(args.reference).name, force=args.force)
    except SpacingMismatch as exc:
        sys.exit(f"error: {exc}")

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
        print(f"-> {args.output}")
        for w in report["warnings"]:
            print(f"warning: {w}")
    else:
        print(text)


if __name__ == "__main__":
    main()
