# ceval

Evaluation of centerline graphs. `ceval` compares a predicted centerline graph with a
reference graph and reports all metrics from the paper under one protocol:

- both graphs are densified so that no edge is longer than the matching tolerance `D`,
  before any metric is computed;
- the tolerance is recorded with every score.

## Quick start (Docker)

```bash
docker build -t ceval .
docker run --rm -v "$PWD/examples:/data" ceval \
    /data/prediction.vtp /data/reference.vtp --d-mm 0.7 -o /data/report.json
```

`examples/` holds a small synthetic vessel tree (`reference.vtp`) and the same tree with one
branch removed (`prediction.vtp`). The report is written to `examples/report.json`.

## Output

By default the report gives one value per metric, grouped as in the paper. Geometric
descriptors (`Len_tot`, `Len_br`, `Ang_bif`, `tortuosity`) and `N_bif` are given as the
absolute difference between prediction and reference. Add `--long-report` to get every
metric with all its details (precision, recall, counts, values of both graphs).

## Your own data

`-o` sets the path of the JSON report; without it the report is printed to the screen. Paths
are inside the container, so mount the folder that holds your graphs, for example as `/work`:

```bash
docker run --rm -v "$PWD:/work" ceval \
    /work/pred.vtp /work/gt.vtp --d-mm 0.7 -o /work/results/case01.json
```

## Input

Two `.vtp` polyline files with node coordinates in millimetres. The tolerance is set with
`--d-mm` (we use the mean voxel diagonal of the image), or with `--dataset` for the datasets
used in the paper (`topcow_mr`, `topcow_ct`, `aortaseg24`, `imagecas`).

## Python

```python
from ceval import evaluate
from ceval.io import read_vtp_polylines

pred_pos, pred_edges = read_vtp_polylines("prediction.vtp")
gt_pos, gt_edges = read_vtp_polylines("reference.vtp")
scores = evaluate(pred_pos, pred_edges, gt_pos, gt_edges, d_mm=0.7)
print(scores["SMD"]["value"], scores["endpoint_F1"]["value"])
```

`evaluate` returns the long form. `ceval.report.summary(ceval.build(...))` gives the short one.

`pred_pos` and `gt_pos` are `(N, 3)` arrays in mm, `pred_edges` and `gt_edges` are `(E, 2)`
arrays of node indices.

## Recommended metrics

We recommend reporting **SMD**, **endpoint F1** and **total length** (`Len_tot`), together with
the structural counts (`beta0_err`, `beta1_err`, `N_bif`, `branch_continuity`) that the clinical
question depends on.

## License

MIT, see [LICENSE](LICENSE).
