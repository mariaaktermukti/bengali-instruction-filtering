# Judge vs human validation (100 annotated rows)

Rows: 100 (Stage 9, Qwen2.5-7B-Instruct judge, the same rubric as Stage 3).

The judge scores on a 1-6 scale; the human labels use 1-5. The judge is mapped onto the human scale with `1 + (judge - 1) * 4/5` and binned to the nearest integer for the agreement statistics. Spearman is unaffected by the map.

| Metric | n | Spearman | p | Kendall τ | Pearson | MAE (1-5 scale) | Exact | ±1 | κ (quad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Complexity | 100 | 0.543 | 0.0000 | 0.430 | 0.537 | 0.717 | 0.400 | 0.950 | 0.316 |
| Quality | 100 | 0.078 | 0.4413 | 0.038 | 0.138 | 1.562 | 0.140 | 0.540 | 0.033 |

## Mean judge score by human score

| Human score | n (complexity) | judge mean | n (quality) | judge mean |
| --- | --- | --- | --- | --- |
| 1 | 22 | 2.20 | 16 | 4.02 |
| 2 | 51 | 2.57 | 25 | 4.27 |
| 3 | 26 | 3.09 | 13 | 4.44 |
| 4 | 1 | 3.38 | 20 | 4.54 |
| 5 | - | - | 26 | 4.20 |

## Score distributions

| Scale | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| human complexity | 22 | 51 | 26 | 1 | 0 | 0 |
| human quality | 16 | 25 | 13 | 20 | 26 | 0 |
| judge complexity (binned) | 0 | 41 | 57 | 2 | 0 | 0 |
| judge quality (binned) | 0 | 0 | 7 | 70 | 23 | 0 |

## Score source

| Source | Rows |
| --- | --- |
| stage9_fresh | 71 |
| stage3_pool | 29 |
