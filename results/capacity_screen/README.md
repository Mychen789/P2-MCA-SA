# Capacity screen — results (manuscript Section 5.3.2, Table 10)

`DEV_EVALUATION.json` is the machine-readable output of the single development
pass over the four cells. Table 10 of the manuscript is a rendering of it.

## Cells

Each cell name is `C<P2><wide>`: the first digit marks the stride-4 P2 head,
the second the wider C3 neck.

| cell | P2 | C3 width | params | AP50 | AP | AP_small | AP_medium | AP_large |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `C00` | no | 96 | 24,116,254 | 39.16 | 22.81 | 18.24 | 36.60 | 49.20 |
| `C01` | no | 192 | 25,862,110 | 39.74 | 23.40 | 18.85 | 37.07 | 42.88 |
| `C10` | yes | 96 | 25,055,336 | 41.32 | 24.23 | 19.89 | 35.96 | 46.16 |
| `C11` | yes | 192 | 27,134,312 | 41.88 | 24.40 | 20.17 | 36.02 | 42.48 |

All values are percentages on the held-out development split, computed with
pycocotools at 768-pixel input.

## Contrasts

In percentage points. `p2_versus_width_matched` is the comparison the screen
exists for: it spends a comparable parameter budget on resolution rather than
on width.

| contrast | AP | AP50 | AP_small | AP_medium | AP_large |
|---|---:|---:|---:|---:|---:|
| P2 at width 96 (C10−C00) | +1.42 | +2.17 | +1.65 | -0.64 | -3.04 |
| P2 at width 192 (C11−C01) | +1.00 | +2.13 | +1.32 | -1.05 | -0.39 |
| width without P2 (C01−C00) | +0.60 | +0.59 | +0.61 | +0.47 | -6.32 |
| width with P2 (C11−C10) | +0.17 | +0.55 | +0.28 | +0.06 | -3.67 |
| **P2 vs width, budget-matched (C10−C01)** | +0.83 | +1.58 | +1.04 | -1.10 | +3.28 |
| interaction | -0.43 | -0.04 | -0.33 | -0.41 | +2.65 |

The P2 cell `C10` carries **806,774 fewer parameters** than the width cell `C01` and
still returns 1.04 points more small-object AP. Capacity alone does not
account for the small-object behaviour of the P2 head in this setting.

## What these numbers are not

- **One seed** (42). No significance claim is available, and none is made.
- **Not comparable to the other tables in this repository.** This is ten-class
  VisDrone at 768 pixels on a sequence-disjoint partition of the official
  training set; the main results are the three-class NMV-SOD task at 960 and
  1280 pixels on the 4555/250/251 split. Resolution, class composition and
  split all differ, so the direction of the P2 effect here and there cannot be
  attributed to any one of them.
- **The large-object column is noise-prone.** It rests on 386 boxes, 0.5% of the
  split, and the numerically largest entries of the contrast table sit in it.

## A note on the paths in the JSON

The `result_root` and `dev_manifest` fields were rewritten on 2026-09-22 to
drop the absolute paths of the machine the screen ran on. `dev_manifest` now
points at the published manifest in this repository, which is byte-identical
to the one that was evaluated: its SHA-256 is recorded in
`../../splits/capacity_screen/dataset_summary.json`. No metric, parameter count
or checkpoint hash was touched.

## Settings recorded in the JSON

- `imgsz`: `768`
- `conf`: `0.001`
- `nms_iou`: `0.7`
- `max_det`: `300`
- `checkpoint_kind`: `"FINAL_EPOCH_EMA_EPOCH_149"`
- `best_by_dev_selection`: `false`
- `official_val_accessed`: `false`
- `bucket_definition`: area after scaling by 768/max(width,height); 32^2 and 96^2 thresholds
