# Derived results

This directory contains compact, derived outputs used to support the manuscript. It does not contain VisDrone imagery, raw annotations, trained weights, or full prediction archives.

- `canonical_3seed.csv`: per-seed AP values for baseline, P2, P2+SA, and P2+MCA+SA.
- `canonical_3seed.json`: the same values with summary statistics, paired tests, and Holm correction over ten prespecified AP/APsmall comparisons.
- `canonical_3seed.md`: publication-ready summary tables.
- `complexity.csv`: parameter, GFLOP, and hardware-dependent throughput measurements.
- `cross_dataset_eval.csv`: same-source held-out evaluation summaries.
- `diagnostics/p2_tal_seed42.json`: TAL audit for P2 without SA.
- `diagnostics/p2_sa_tal_seed42.json`: matched TAL audit with SA (`HI_RATIO=32`).

AP values in the canonical files are percentages. Standard deviations are sample standard deviations across seeds 42, 1, and 7.

