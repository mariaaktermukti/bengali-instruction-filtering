# Kaggle LoRA experiments

LoRA fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct` on each data-selection arm from the
Iteration 03 pilot, run on Kaggle (one Tesla T4). Results are in [`results/lora_arms/`](results/lora_arms/).

> The Kaggle notebooks are **private** to the `habiburrahman153` account. Ask for access,
> or rerun them with the steps below.

## Notebooks

| Notebook | What it does | Output |
| --- | --- | --- |
| [bangla-filtering-selection](https://www.kaggle.com/code/habiburrahman153/bangla-filtering-selection) | Iteration 03 pipeline: filters, dedup, decontamination, LLM judge (c, q), IFD, embeddings, arms A–F | `arms/arm_*.jsonl`, `arms/heldout.jsonl` |
| [bengali-lora-arm-a](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-a) | LoRA on arm A (whole clean pool, 955 rows) | adapter + `training_report.json` |
| [bengali-lora-arm-b](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-b) | LoRA on arm B (random 400) | adapter + `training_report.json` |
| [bengali-lora-arm-c](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-c) | LoRA on arm C (c × q + diversity, 400) | adapter + `training_report.json` |
| [bengali-lora-arm-d](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-d) | LoRA on arm D (top q, 400) | adapter + `training_report.json` |
| [bengali-lora-arm-f](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-f) | LoRA on arm F (top IFD < 1, 400) | adapter + `training_report.json` |
| [bengali-lora-heldout-eval](https://www.kaggle.com/code/habiburrahman153/bengali-lora-heldout-eval) | Per-example held-out loss for the base model and all five adapters | `heldout_per_example.csv` |
| [bengali-toy-lora](https://www.kaggle.com/code/habiburrahman153/bengali-toy-lora) | Debugging runs of the original toy script (v1 torchao crash, v2 smoke test, v3 first arm C run) | – |

Arm E was not trained: it is the same 400 rows, in the same order, as arm C, because the
diversity threshold τ = 0.9 removed nothing from this pool.

## Results

Held-out set: 47 rows, mean response-token cross-entropy. Base model before training: **0.958**.

| Arm | Rows | Steps | Held-out loss | Change vs base | vs random (B), 95% CI | Minutes |
| --- | --- | --- | --- | --- | --- | --- |
| A | 955 | 239 | **0.881** | −0.077 | −0.020 [−0.025, −0.016] | 13.5 |
| B | 400 | 100 | 0.901 | −0.057 | – | 5.4 |
| C | 400 | 100 | 0.902 | −0.056 | +0.001 [−0.002, +0.003] | 5.8 |
| D | 400 | 100 | 0.907 | −0.051 | +0.005 [−0.0001, +0.013] | 5.9 |
| F | 400 | 100 | 0.910 | −0.048 | +0.008 [+0.002, +0.016] | 6.4 |

All arms improve on the base model. No 400-row selection rule beats random selection;
IFD selection (F) is worse than random. Using all 955 rows beats every 400-row subset.

Files:

- [`summary.csv`](results/lora_arms/summary.csv): one row per arm
- [`heldout_comparison.csv`](results/lora_arms/heldout_comparison.csv): paired bootstrap against the base model and arm B
- [`heldout_by_domain.csv`](results/lora_arms/heldout_by_domain.csv): held-out loss by task domain
- [`heldout_per_example.csv`](results/lora_arms/heldout_per_example.csv): per-example losses
- `arm_*_training_report.json`: config, token stats, loss curve, before/after generations

## Fixes needed to run on Kaggle

The original [`scripts/toy_lora_training.py`](scripts/toy_lora_training.py) does not run on Kaggle.
[`scripts/kaggle_lora_training.py`](scripts/kaggle_lora_training.py) fixes:

1. **torchao crash.** Kaggle ships `torchao` 0.10.0; current `peft` raises
   `ImportError: Found an incompatible version of torchao` in `get_peft_model`.
   The script uninstalls `torchao` (LoRA does not use it).
2. **Training data not in git.** `data/` is gitignored. The script finds `arm_<X>.jsonl` and
   `heldout.jsonl` anywhere under `/kaggle/input` (attach the selection notebook's output).
3. **512-token limit.** Qwen needs ~1,160 tokens for the median Bengali row; 512 truncated
   351 of 400 rows. Now 3,072 tokens with gradient checkpointing (peak 7.6 GB on a T4).
4. **Two GPUs.** "T4 x2" made `Trainer` use `DataParallel`; the script pins one GPU.
5. **Precision.** New `transformers` loads Qwen in bf16, which a T4 only emulates. The script
   loads fp32 weights and trains with fp16 mixed precision.
6. **Prompt format.** Uses Qwen's chat template and computes loss on the response only.

## Rerunning

Requirements: Kaggle CLI logged in, phone-verified account (for Internet and GPU).

```bash
# train arms B, C, D, F, A one after another, save reports to results/lora_arms/
bash kaggle/run_arms.sh b c d f a

# per-example held-out loss: push scripts/heldout_per_example.py as a Kaggle script
# with the five arm notebooks + bangla-filtering-selection attached, then:
python scripts/compare_arms_bootstrap.py results/lora_arms/heldout_per_example.csv
```

Settings shared by all arms: LoRA r = 8, α = 16, dropout 0.05 on q/k/v/o; 1 epoch;
learning rate 1e-4; batch 1 × 4 gradient accumulation; seed 42.
