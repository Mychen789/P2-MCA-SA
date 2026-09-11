# Capacity screen — frozen split (manuscript Section 5.3.2)

The partition used by the 2×2 capacity control. It divides the **official
VisDrone ten-class training set** into a training part and a held-out
development part. It is not the NMV-SOD-3cls split used everywhere else in
this repository; see `../README.md` for that one.

| | images | sequences |
|---|---:|---:|
| `phaseb1_train_manifest.tsv` | 5,149 | 166 |
| `phaseb1_dev_manifest.tsv` | 1,322 | 42 |
| total | 6,471 | 208 |

**The two parts share no video sequence** (0 in common). VisDrone frames
inside one sequence are near-duplicates, so an image-level split would have
leaked development frames into training in all but name. The totals above sum
to the whole official training set, so no development image was trained on.

## Provenance

`freeze_commit.json` records the state of the split when it was committed:

- `status`: `FROZEN_BEFORE_MODEL_OUTPUT`
- `training_used`: `false`
- `model_forward_used`: `false`
- `effect_direction_viewed`: `false`

That is, the split was fixed before any model ran on it and before any effect
direction had been seen. The same file carries the SHA-256 of each manifest,
which is how that claim stays checkable. Recomputed from the files as
published:

| file | recorded SHA-256 | recomputed | match |
|---|---|---|---|
| `phaseb1_train_manifest.tsv` | `02E74828C805C9BF…` | `02E74828C805C9BF…` | yes |
| `phaseb1_dev_manifest.tsv` | `6BE3E7B697AD6A9A…` | `6BE3E7B697AD6A9A…` | yes |

## Columns

One row per image. `phaseb1_role` is `TRAIN` or `DEV`; `image_rel` and
`label_rel` are paths under the VisDrone YOLO-format root; `image_sha256` and
`label_sha256` pin the exact file contents; `class_counts` is the ten-class
histogram; `size_counts_768` is the small/medium/large histogram under the
same definition the evaluation uses — box area after scaling by
`768 / max(width, height)`, split at `32²` and `96²`.

The images and annotations themselves are **not** redistributed here. They
come from the public VisDrone2019-DET benchmark.

## Development split composition

1,322 images, 74,642 effective boxes: 63,370 small, 10,886 medium, 386 large.

The large bucket is 0.5% of the boxes, which is why the manuscript does not
interpret the large-object column of Table 10.
