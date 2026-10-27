![cEval](img_readme.jpeg)

# ceval

Evaluation of centerline graphs. `ceval` compares a predicted centerline graph with a
reference graph and reports all metrics from the paper under one protocol:

- the curve metrics compare the curves themselves, so they do not depend on where either
  graph places its nodes;
- the node metrics (`node_F1`, `chamfer`, `false_merges`, `false_splits`) depend on node
  spacing and are reported separately;
- the tolerance is recorded with every score.

![No single metric reports every error](docs/overview.png)

*Different metrics respond to different errors (blue: the metric responds, red: it does not).
No single metric reports every error, which is why the choice of metrics matters.*

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
descriptors (`Len_tot`, `Len_br`, `Ang_bif`, `tortuosity`) are given as the absolute
difference between prediction and reference. Add `--long-report` to get every
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

Run from the repository root:

```python
from ceval import build
from ceval.io import read_vtp_polylines
from ceval.report import summary

pred_pos, pred_edges = read_vtp_polylines("examples/prediction.vtp")
gt_pos, gt_edges = read_vtp_polylines("examples/reference.vtp")

report = build(pred_pos, pred_edges, gt_pos, gt_edges, tau=0.7)   # long report
short = summary(report)                                            # one value per metric

for group, metrics in short["metrics"].items():
    print(group)
    for name, value in metrics.items():
        print(f"  {name:<20s} {value}")
```

Output:

```
coverage
  edge_F1              0.9159
  edge_R               0.8448
  overlap_OV           0.9154
  branch_coverage_F1_80 0.9231
  branch_coverage_F1   0.8766
distance
  ASCD                 0.4183
  HD95                 4.5151
  netmets_geom_F1      0.9176
path
  APLS                 0.8207
topology
  junction_F1          0.8
  endpoint_F1          0.8889
  netmets_conn_F1      1.0
count
  beta0_err            0.0
  beta1_err            0.0
  branch_count_ratio   0.2857
nodes
  node_F1              0.9114
  chamfer              0.4379
  false_merges         0.0
  false_splits         0.0
descriptor
  Len_tot              14.1421
  Len_br               2.0152
  Ang_bif              0.0
  tortuosity           0.0
```

`pred_pos` and `gt_pos` are `(N, 3)` arrays in mm, `pred_edges` and `gt_edges` are `(E, 2)`
arrays of node indices. `report` holds the long report (the same as `--long-report`).

## Recommended metrics

We recommend reporting $\color{red}{\textsf{\textbf{ASCD} (average symmetric centerline distance, Liu et al. 2025)}}$ [[doi]](https://doi.org/10.1002/mp.17855), **endpoint F1** and **total length** (`Len_tot`), together with
the structural counts (`beta0_err`, `beta1_err`, `branch_count_ratio`) that the clinical
question depends on.

## License

MIT, see [LICENSE](LICENSE).
