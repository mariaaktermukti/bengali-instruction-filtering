# Blind spot-check: arm B (random 10K) vs arm C (DEITA, c x q + diversity)

Items: 100 (50 per arm, arm hidden in the sheet). Source: Stage 4 `reports/spotcheck_blind.csv`, key `reports/spotcheck_key.csv`.

## Qwen judge (the model that selected the arms - circular for this comparison)

| Arm | n | mean complexity | mean quality | mean c x q | c x q 95% CI |
| --- | --- | --- | --- | --- | --- |
| B | 50 | 3.626 | 4.885 | 17.605 | [16.323, 18.834] |
| C | 50 | 4.510 | 5.178 | 23.250 | [22.873, 23.617] |

## Independent rater (non-circular)

Rater: `unsloth/Llama-3.1-8B-Instruct` - a different model family from the Qwen judge that selected the arms, so this comparison does not reward the selection criterion itself. Two independent samples: 50 C items vs 50 B items.

| Arm | n | mean rating | 95% CI |
| --- | --- | --- | --- |
| B | 50 | 4.055 | [4.005, 4.101] |
| C | 50 | 4.124 | [4.088, 4.164] |

Mean C - mean B = **+0.070** (95% CI [+0.009, +0.133], p(C better) = 0.990).

## Human ratings (blind)

_No human ratings yet. Fill the `item` / `rating` / `notes` columns of `results/full_run/spotcheck_ratings.csv` and re-run this script._

