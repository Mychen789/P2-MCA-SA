# E26 seed 7 manuscript backfill

## Table 6 ready rows (%, mean±sample SD; APlarge shown as mean)

| Configuration | AP50 | AP | APsmall | APmedium | APlarge |
|---|---:|---:|---:|---:|---:|
| Baseline | 53.68±0.27 | 28.94±0.36 | 24.47±0.35 | 42.81±0.58 | 35.87 |
| +P2 | 52.78±1.08 | 28.46±0.44 | 24.21±0.43 | 41.91±0.49 | 35.64 |
| +P2+SA | 52.77±0.06 | 28.63±0.27 | 24.31±0.22 | 41.96±0.59 | 26.71 |
| +P2+MCA+SA | 53.36±0.34 | 29.61±0.49 | 25.09±0.26 | 43.05±0.90 | 32.41 |

## Paired tests for the ten pre-specified AP/APsmall comparisons

| Contrast | Metric | Δ (pp) | paired t | raw p | Holm-adjusted p |
|---|---|---:|---:|---:|---:|
| P2 vs baseline | AP | -0.480 | -2.425 | 0.1362 | 0.6808 |
| P2+SA vs P2 | AP | +0.171 | 0.435 | 0.7057 | 1.0000 |
| Full vs P2+SA | AP | +0.980 | 2.269 | 0.1513 | 0.6808 |
| Full vs P2 | AP | +1.151 | 12.971 | 0.0059 | 0.0589 |
| Full vs baseline | AP | +0.670 | 4.423 | 0.0475 | 0.3800 |
| P2 vs baseline | APsmall | -0.263 | -1.034 | 0.4096 | 1.0000 |
| P2+SA vs P2 | APsmall | +0.102 | 0.349 | 0.7607 | 1.0000 |
| Full vs P2+SA | APsmall | +0.782 | 3.271 | 0.0821 | 0.4926 |
| Full vs P2 | APsmall | +0.884 | 7.715 | 0.0164 | 0.1475 |
| Full vs baseline | APsmall | +0.621 | 4.032 | 0.0564 | 0.3945 |

## Required manuscript synchronization

1. Add the +P2+SA row to Table 6.
2. Change the six-comparison Holm family to ten comparisons.
3. Report P2+SA vs P2 as the isolated SA contrast and Full vs P2+SA as MCA conditional on SA.
4. Update Figure 5(a,b) to include P2+SA, keeping per-seed points and no significance stars.
5. Remove the limitation statement that 1280-pixel P2+SA multi-seed confirmation is unfinished.
