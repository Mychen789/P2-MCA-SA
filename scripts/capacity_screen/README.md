# Capacity screen — code (manuscript Section 5.3.2)

The 2×2 factorial control reported in Section 5.3.2 and Table 10 of the
manuscript: the stride-4 P2 head (on/off) crossed with C3 neck width (96/192),
four cells trained under one frozen protocol and evaluated once on a held-out
development split.

## Read this before running anything

**These files are a record of what was executed, not a turnkey package.** They
carry absolute paths from the machine the screen ran on:

- `D:/visdrone10_yolo` as the dataset root (`a1_phaseb1r_engine.py`);
- `C:/Users/cmy/anaconda3/envs/yolov8/Lib/site-packages/ultralytics/cfg/models/v8/yolov8.yaml`
  as the stock model YAML (`a1_phaseb1r_numeric.py`);
- project-relative paths derived from the original directory layout.

Reproducing the screen elsewhere means editing those constants. We publish the
code unedited so that what is shown here is exactly what produced Table 10,
rather than a cleaned-up version that was never run.

`a1_phaseb1r_numeric.py` additionally verifies the SHA-256 of
`a1_phaseb0_builder.py` and `a1_phaseb0_data.py` before importing them; if you
edit those two files, the recorded digests will no longer match and the import
gate will refuse to proceed. That gate is deliberate — it is what made the
frozen-architecture claim checkable — so change the digests consciously.

## Files

| File | Role |
|---|---|
| `b1r_config.json` | The frozen experiment configuration: cell order, seed 42, 768-pixel input, batch 2, 150 epochs, SGD, cosine schedule, no early stopping, no training-time validation, `AP_S_768` as primary metric |
| `a1_phaseb0_builder.py` | Builds the four cells from the P2 model YAML and transfers the common pretrained body; the source of the parameter counts in Table 10 |
| `a1_phaseb0_data.py` | Dataset adapter that feeds an explicit image list to the trainer |
| `a1_phaseb1r_contract.py` | Cell identities, frozen training fields, and the transaction rules that keep development evaluation locked until four training commits validate |
| `a1_phaseb1r_engine.py` | Reads and validates the frozen manifests, verifying every image by byte count and SHA-256 |
| `a1_phaseb1r_cell.py` | Trainer policy: locked batch, optimiser coverage checks, EMA handling, and the final-epoch checkpoint with its strict-reload QC |
| `a1_phaseb1r_numeric.py` | Assembles the numeric stack and builds one attested cell |
| `b1r_runner.py` | Runs or resumes exactly one cell per invocation and reports the single allowed next action |
| `a1_phaseb1r_dev_eval.py` | The development evaluation described below |

## How the screen was run

Training: `b1r_runner.py` was invoked once per cell, in the frozen order
C10, C11, C01, C00. Each invocation trains or resumes exactly one cell and then
stops; the runner derives the next allowed action from the `status.json` files
rather than from any argument, so the cell order cannot be changed by mistake.

Evaluation: after all four cells reached `COMPLETE`, `a1_phaseb1r_dev_eval.py`
was run once. It is the adapter the runner reports as *"four-cell evaluation
adapter is not connected yet"* — the original design left the evaluation step
unimplemented, and it was written on 2026-09-11, after training finished and
before any effect direction had been read.

What that evaluation does, and deliberately does not do:

- it loads only the four `C**_FINAL_EPOCH_EMA_EPOCH_149.pt` checkpoints, never
  `best.pt` or `last.pt`, so no model selection on development data occurs;
- it rebuilds each architecture from the builder and checks the parameter total,
  the strides, the EMA state hash and a strict `load_state_dict` before
  inference;
- it reads only the DEV rows of the frozen manifest, verifying each image by
  byte count and SHA-256; the official VisDrone validation and test sets are
  never opened;
- its size buckets reproduce the frozen manifest definition — box area after
  scaling by `768 / max(width, height)`, split at `32²` and `96²` — so
  `AP_S_768` means the same thing here as `size_counts_768` does in the data
  freeze;
- detection settings match `scripts/eval_size_buckets.py`: `imgsz=768`,
  `conf=0.001`, NMS `iou=0.7`, COCO AP through pycocotools.

It writes `DEV_EVALUATION.json`, published under `results/capacity_screen/`.

## Scope

One seed. The screen carries no significance claim, and the manuscript labels it
as single-seed wherever it is cited. Its large-object column rests on 386 boxes
and is not interpreted.
