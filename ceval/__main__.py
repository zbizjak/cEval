"""cEval CLI: two centerline graphs in, one JSON report out.

    python3 -m ceval pred.vtp gt.vtp --d-mm 0.7
    python3 -m ceval pred.vtp gt.vtp --dataset topcow_mr -o report.json
    python3 -m ceval pred.vtp gt.vtp --d-mm 0.7 --long-report
"""
import argparse
import json
from pathlib import Path

from ceval.io import read_vtp_polylines
from ceval.report import build, summary

# Tolerance tau (dataset mean voxel diagonal, mm) used in the paper.
# mean voxel diagonal rounded to 0.1 mm (see janniks/lib/datasets.py)
D_MM = {"topcow_mr": 0.7, "topcow_ct": 1.0, "aortaseg24": 1.6, "imagecas": 0.7}


def main():
    ap = argparse.ArgumentParser(prog="ceval", description=__doc__)
    ap.add_argument("prediction", help="predicted graph (.vtp), mm coordinates")
    ap.add_argument("reference", help="reference graph (.vtp), mm coordinates")
    ap.add_argument("--d-mm", type=float,
                    help="tolerance tau in mm (mean voxel diagonal)")
    ap.add_argument("--dataset", help="look D up by dataset name instead")
    ap.add_argument("-o", "--output", help="write here (default: stdout)")
    ap.add_argument("--long-report", "--long_report", dest="long_report",
                    action="store_true",
                    help="write every metric with all its details "
                         "(default: one value per metric)")
    args = ap.parse_args()

    if args.d_mm is None:
        if not args.dataset:
            ap.error("one of --d-mm or --dataset is required")
        if args.dataset not in D_MM:
            ap.error(f"unknown dataset {args.dataset!r}; known: {sorted(D_MM)}")
        args.d_mm = D_MM[args.dataset]

    pred = read_vtp_polylines(args.prediction)
    gt = read_vtp_polylines(args.reference)
    report = build(*pred, *gt, args.d_mm,
                   pred_name=Path(args.prediction).name,
                   gt_name=Path(args.reference).name)

    if not args.long_report:
        report = summary(report)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
        print(f"-> {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
