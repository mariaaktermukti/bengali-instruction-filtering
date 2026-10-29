# HANDOFF: full 100K run on Kaggle

Read this first in a new session. Step tracker to the paper: [ROADMAP.md](ROADMAP.md). Plan and reasoning: [kaggle/full_run/PLAN.md](kaggle/full_run/PLAN.md).
Update the **Status** table and **Next action** after every step, and add a line to the log.

## Status

| Stage | Kernel (habiburrahman153/...) | State | Output / key numbers |
| --- | --- | --- | --- |
| 1 clean (CPU) | `bn-full-s01-clean` | **DONE** (v1, ~11 min) | 100K sampled -> **91,120 clean pool**, **470 held-out**. Drops: heuristics 5,081 (language 4,215), exact dups 2,991, near dups 736, decontamination 71 (62 held-out, 8 MMLU-bn). 111.5M Llama tokens |
| 2 IFD + embeddings (2×T4) | `bn-full-s02-ifd-embed` | **DONE** (v1, 223 min: IFD 192, embed 30.5) | IFD for all 91,120 rows (IFD ≥ 1: 0.02%). Median IFD by domain 0.75 (code) to 0.88. bge-m3 sanity: random median 0.38, p99 0.58; MinHash near-dups median 0.98, 100% above random p99. **NN-sim quantiles** p10 .680, p25 .746, p50 .815, p75 .883, p90 .939, p95 .959, p99 .978 |
| 2b IFD with Qwen (2×T4) | `bn-full-s02b-ifd-qwen` | **RUNNING** (v1 pushed 2026-10-30 00:19 BST; budget 1.75 h) | IFD recomputed with **Qwen2.5-0.5B-Instruct**, the model Stage 5 trains (Cherry-LLM scores with the tuned model). Chat-template prompt as in `kaggle_lora_training.py`, fp32 + fp16 autocast. Output `scores/ifd_qwen.parquet` + `ifd_qwen_summary.json` (incl. Spearman vs the Llama IFD). Last GPU job on account habiburrahman153 (2.5 h quota left) |
| 3a judge benchmark (2×T4) | `bn-full-s03a-judge-bench` | **DONE** (v2, 62 min; v1 was partial) | vLLM 0.32.0 works after uninstalling torchaudio (torch 2.13+cu130, cuda True, 2 GPUs). Rows/h and hours for 91,120 rows, Spearman c vs pilot / MAE: `hf` 2,763 (33.0 h, 0.9999 / 0.003); **`vllm_tp2` 4,374 (20.8 h, 1.0000 / 0.0027)**; `awq_dp2` 4,814 (18.9 h, 0.962 / 0.20, q 0.923); `gptq8_dp2` 4,243 (21.5 h, 0.999 / 0.014). Parse 1.00 everywhere, max prompt 3,131 tok. Script `recommended` = `awq_dp2`, but the rule is max rows/h with no quality check. **Use `vllm_tp2`**: AWQ is only 10% faster and changes the ranking (MAE 0.20) |
| 3 judge (2×T4) | `bn-full-s03-judge-01`, `-02`, ... | **S01 DONE** (623 min); **S02 DONE** (629 min, 90,000 / 91,120); **S03 DONE** (24 min) → **Stage 3 COMPLETE: 91,120 / 91,120, parse 1.00** | S01: **46,000 / 91,120** scored, 4,458 rows/h, parse 1.00, clean stop at the time budget, no crash or eager retry. Startup 9 min, 26.7 min per chunk. S02: +44,000 -> **90,000 / 91,120**, 4,244 rows/h (28 min per chunk, 5% slower than S01), parse 1.00, stopped at the budget **1,120 rows short**. Merged parquet checked: 90,000 unique ids, no nulls, S01 scores identical; S02's new rows have the same c/q distributions. S03 needs ~15 min scoring + ~9 min startup. On the S01 rows, c: median 4.00, p10 2.18, p90 4.79. **q has a ceiling**: median 5.00, 71% ≥ 4.9, but 18% < 4 and 7% < 3. Spearman(c, q) = −0.23. Median c by domain: translation 2.23, mcq 2.78, other 2.95, the rest 3.7–4.0 |
| 4 select arms (CPU) | `bn-full-s04-select` | **v3 DONE (τ 0.74, Llama IFD)**; **v4 to push** after 2b (arm F with Qwen IFD) | v1 τ 0.90, v2 sweep 0.65–0.96, v3 τ 0.74. Arms A 91,120, B–F 10,000; held-out 470. v4 changes arm F only (A–E use no IFD). Reports in `outputs/full/s04/` (v1 arms kept in `outputs/full/s04_v1_tau090/`, stale). τ 0.74 ≈ pool NN-sim p25 (0.746): **C∩E 58%** (v1: 91.5%), C depth 18,768. Diagnostics C / E / B: mean_pair_sim .379 / .391 / .386, distinct-2 .503 / .451 / .515, coverage .746 / .727 / .757, c_mean 4.49 / 4.74 / 3.71. Code share: pool 11.4%, C 3.0%, E 1.4%. Tokens: B 12.2M, C 15.5M, D 15.5M, E 15.2M, F 13.3M. D cut q ≥ 5.44 (no ceiling problem) |
| 5 LoRA per arm | - | not written; runs on the **new Kaggle account** | base model **Qwen2.5-0.5B-Instruct** |
| 6 eval | - | not written | - |

Stage 1 caveat: the filter-bias check flagged **code (20.7% dropped, mostly `language`: 3,038 rows)** and
**mcq_classification (21.0%, mostly `translationese`: 366 rows)**. The pilot showed the same pattern (code 19%).
Likely cause: code outside ``` fences counts as English prose. The held-out set is filtered the same way, so the
comparison between arms stays fair, but say so in the thesis, or loosen the rule and rerun Stage 1 before Stage 3.

## Next action

Account `habiburrahman153` has no GPU quota left after Stage 2b (it had 2.5 h on 2026-10-30; the quota resets weekly).
CPU kernels (Stage 4) do not use GPU quota and can still run there. **All GPU work from Stage 5 on runs on a second account.**

1. **Check Stage 2b** (expected ~1.5–1.9 h after the push):
   ```bash
   export PYTHONUTF8=1
   kaggle kernels status habiburrahman153/bn-full-s02b-ifd-qwen
   kaggle kernels output habiburrahman153/bn-full-s02b-ifd-qwen -p outputs/full/s02b
   cat outputs/full/s02b/bangla_filtering/full/scores/ifd_qwen_summary.json
   ```
   Expect `ifd_scored` 91,120 and `nan_or_missing` 0. If it stopped short (the log says INCOMPLETE), the rows left must be
   finished on the new account: upload `ifd_qwen.parquet` into a dataset there (keep the path `full/scores/ifd_qwen.parquet`),
   attach it plus the pool, and rerun `s02b` (it resumes). If many rows are NaN, fp16 overflowed; then use the Llama IFD for arm F
   and note the mismatch in the paper.
2. **Stage 4 v4** (CPU, still on habiburrahman153; `kernel-metadata.json` already attaches s02b; Stage 4 uses the Qwen IFD
   when `ifd_qwen.parquet` is attached, prints `IFD source`):
   ```bash
   kaggle kernels push -p kaggle/full_run/s04_select
   kaggle kernels status habiburrahman153/bn-full-s04-select
   kaggle kernels output habiburrahman153/bn-full-s04-select -p outputs/full/s04_v4     # ~830 MB with the arms
   ```
   Check `reports/stage4_manifest.json`: `ifd_source` = Qwen, τ 0.74, C∩E 58% as in v3 (A–E must not change),
   and `judge_report.spearman_ifd_vs_ifd_llama`.
3. **Move to the second account.** A private kernel's output cannot be attached from another account, so the arms go over as a
   private dataset:
   1. On the new account: verify the phone number (needed for GPU and Internet), Settings → API → create a token.
   2. Locally: keep the old token (`mv ~/.kaggle/access_token ~/.kaggle/access_token.habiburrahman153`), save the new one as
      `~/.kaggle/access_token`, and check with `kaggle kernels list --mine`. Switch back the same way for account-1 jobs.
   3. Build the dataset from the v4 output (arm A is 506 MB; include it only if arm A will be trained):
      ```bash
      mkdir -p outputs/full/arms_dataset && cp outputs/full/s04_v4/bangla_filtering/full/arms/{arm_B,arm_C,arm_D,arm_E,arm_F,heldout}.jsonl outputs/full/arms_dataset/
      cp outputs/full/s04_v4/bangla_filtering/full/reports/stage4_manifest.json outputs/full/arms_dataset/
      kaggle datasets init -p outputs/full/arms_dataset      # edit dataset-metadata.json: id "<NEW_USER>/bn-full-arms", title "bn-full-arms"
      kaggle datasets create -p outputs/full/arms_dataset    # private by default
      ```
   4. Record the new username in this file (Conventions) and in `kernel-metadata.json` ids from Stage 5 on.
4. **Stage 5 (LoRA, new account).** Base model **Qwen2.5-0.5B-Instruct** (decided). Adapt `scripts/kaggle_lora_training.py` into
   `kaggle/full_run/s05_lora/`: two arms per session (one per GPU), `dataset_sources: ["<NEW_USER>/bn-full-arms"]`,
   checkpoint + resume, time budget. Order: B + C first (main comparison), then D + E, then F. ~2.4 h per arm (pilot rate).
5. Report in the paper: token totals differ per arm (C +27% vs B), so compare per token as well as per row; code is under-selected by c × q.

## Decisions

- Arm size **K = 10,000** for arms B–F (user, 2026-10-26). Set `SUBSET_SIZE = 10_000` in Stage 4.
- **τ = 0.74** (2026-10-30): the notebook's 0.90 left C 91.5% identical to E; 0.74 ≈ pool NN-sim p25, C∩E 58%.
- **Base model for LoRA: Qwen2.5-0.5B-Instruct** (user, 2026-10-30): ~2.4 h per 10K arm vs ~5–6 h for Llama-3.2-1B; IFD recomputed with it (Stage 2b) so arm F matches.
- **Kaggle accounts** (2026-10-30): stages 1–4 on `habiburrahman153`; GPU work from Stage 5 on a second account (name TBD).
- Arm A (whole pool, 91,120 rows, ~22 h with Qwen-0.5B): still open. Options: multi-session training with resume, or a compute-matched run.

## Conventions

- Accounts: `habiburrahman153` (stages 1–4, GPU quota used up 2026-10-30) and `<NEW_USER>` (Stage 5+; fill in). Kernel ids in
  `kernel-metadata.json` carry the account name; a kernel can only attach outputs of the same account (or public ones / datasets).
- Each stage lives in `kaggle/full_run/sNN_<name>/` (script + `kernel-metadata.json`) and is pushed with
  `kaggle kernels push -p <dir>`. Run the kaggle CLI with `PYTHONUTF8=1`, because Bengali logs break cp1252 on Windows.
- Outputs go under `/kaggle/working/bangla_filtering/full/{pool,scores,arms,reports}`, the same layout as the pilot notebook.
  Download them into `outputs/full/sNN/` (gitignored).
- `kaggle/full_run/s01_clean/s01_clean.py` is generated from the notebook
  (`kaggle/notebooks/bangla-filtering-selection.ipynb`, cells 2–16), so the filters match the pilot byte for byte.
  If you change the filters, change the notebook too.
- GPU stages: `machine_shape: NvidiaTeslaT4` = 2 × T4. Keep both busy. Stop on a time budget (10.5 h)
  and checkpoint, because a session killed at 12 h loses its output.

## Log

- 2026-10-26: pulled the selection notebook from Kaggle into `kaggle/notebooks/`. Pilot timings: judge 3,216 s per 955 rows
  (4 prompts per row), IFD 207 s, embeddings 18 s. Generated and pushed Stage 1; wrote the Stage 2 script.
- 2026-10-26: Stage 1 done (91,120 clean / 470 held-out). Pushed Stage 2. User chose K = 10,000.
- 2026-10-27: Stage 2 done (223 min, full IFD coverage). Wrote and pushed Stage 3a (judge benchmark: hf / vllm_tp2 / awq_dp2 / gptq8_dp2).
- 2026-10-27: Stage 3a v1 done (15.5 min). hf: 2,499 rows/h (36.5 h for the pool), matches the pilot (Spearman 0.9999). All vLLM backends failed (torchaudio cu128 vs torch cu130 after `pip install vllm`). Wrote the v2 fix (uninstall torchaudio) and pushed v2.
- 2026-10-27: Stage 3a v2 done (62 min). vLLM works. vllm_tp2 4,374 rows/h (20.8 h), exact vs pilot; awq_dp2 fastest (18.9 h) but Spearman 0.95 vs hf and MAE 0.20, so Stage 3 uses vllm_tp2.
- 2026-10-27: wrote Stage 3 (`s03_judge.py`: vLLM 0.32.0 TP=2, pinned; shuffled row order, so a partial run is a random sample; checkpoint every 2K rows; worker retried in eager mode on a crash; timeout before the 12 h kill). Tested crash and resume locally with a fake vLLM on the 955 pilot rows. Pushed session 01.
- 2026-10-28: Stage 3 session 01 done (623 min): 46,000 / 91,120 rows, 4,458 rows/h, parse 1.00, clean stop at the time budget. q has a ceiling (71% ≥ 4.9). Pushed session 02.
- 2026-10-29: Stage 3 session 02 done (629 min): 90,000 / 91,120, 4,244 rows/h, parse 1.00, stopped 1,120 rows short at the budget. Merged parquet verified (no dups/nulls, S01 scores unchanged, same distributions). Pushed session 03.
- 2026-10-29: wrote Stage 4 (`s04_select.py`, CPU): notebook cells 22–31 + τ sweep; tested on synthetic data (greedy = naive reference). Not pushed until S03 is done.
- 2026-10-29: Stage 3 session 03 done (24 min): **91,120 / 91,120 scored**, parse 1.00; final parquet verified (`outputs/full/s03_03/`). Pushed Stage 4 v1.
- 2026-10-29: Stage 4 v1 done (1.2 min, τ 0.90). C∩E 91.5%; τ sweep 0.80–0.96 gives 76–97% overlap. Code share 11.4% → 1.6% in C/E. D cut q ≥ 5.44 (no ceiling problem). τ decision open.
- 2026-10-30: wrote ROADMAP.md (37 steps to the paper, 10 done → 11 with τ). Stage 4 v2: τ sweep 0.65–0.96 (C∩E 29% at 0.65 → 97% at 0.96). Stage 4 v3 with **τ 0.74**: C∩E 58%, C now the most diverse arm. Stage 4 DONE.
- 2026-10-30: user chose **Qwen2.5-0.5B-Instruct** for LoRA. Wrote Stage 2b (`s02b_ifd_qwen.py`, IFD with Qwen, budget 1.75 h for the 2.5 h of quota left) and pushed v1. Stage 4 now prefers `ifd_qwen.parquet` for arm F (tested both paths on synthetic data). Planned the switch to a second Kaggle account for Stage 5+.
