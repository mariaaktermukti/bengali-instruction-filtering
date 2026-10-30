# HANDOFF: full 100K run on Kaggle

Read this first in a new session. Step tracker to the paper: [ROADMAP.md](ROADMAP.md). Plan and reasoning: [kaggle/full_run/PLAN.md](kaggle/full_run/PLAN.md).
Update the **Status** table and **Next action** after every step, and add a line to the log.

## Status

| Stage | Kernel (habiburrahman153/...) | State | Output / key numbers |
| --- | --- | --- | --- |
| 1 clean (CPU) | `bn-full-s01-clean` | **DONE** (v1, ~11 min) | 100K sampled -> **91,120 clean pool**, **470 held-out**. Drops: heuristics 5,081 (language 4,215), exact dups 2,991, near dups 736, decontamination 71 (62 held-out, 8 MMLU-bn). 111.5M Llama tokens |
| 2 IFD + embeddings (2×T4) | `bn-full-s02-ifd-embed` | **DONE** (v1, 223 min: IFD 192, embed 30.5) | IFD for all 91,120 rows (IFD ≥ 1: 0.02%). Median IFD by domain 0.75 (code) to 0.88. bge-m3 sanity: random median 0.38, p99 0.58; MinHash near-dups median 0.98, 100% above random p99. **NN-sim quantiles** p10 .680, p25 .746, p50 .815, p75 .883, p90 .939, p95 .959, p99 .978 |
| 2b IFD with Qwen (2×T4) | `bn-full-s02b-ifd-qwen` | **DONE** (v1, 101 min) | IFD with **Qwen2.5-0.5B-Instruct** (the model Stage 5 trains) for all 91,120 rows, **0 NaN** (fp32 + fp16 autocast). IFD ≥ 1: 4.7% (Llama: 0.02%), median 0.83. **Spearman vs the Llama IFD 0.68**, so arm F changes a lot. Median by domain: code 0.94 (Llama 0.75), reading_comp 0.71. Output in `outputs/full/s02b/` |
| 3a judge benchmark (2×T4) | `bn-full-s03a-judge-bench` | **DONE** (v2, 62 min; v1 was partial) | vLLM 0.32.0 works after uninstalling torchaudio (torch 2.13+cu130, cuda True, 2 GPUs). Rows/h and hours for 91,120 rows, Spearman c vs pilot / MAE: `hf` 2,763 (33.0 h, 0.9999 / 0.003); **`vllm_tp2` 4,374 (20.8 h, 1.0000 / 0.0027)**; `awq_dp2` 4,814 (18.9 h, 0.962 / 0.20, q 0.923); `gptq8_dp2` 4,243 (21.5 h, 0.999 / 0.014). Parse 1.00 everywhere, max prompt 3,131 tok. Script `recommended` = `awq_dp2`, but the rule is max rows/h with no quality check. **Use `vllm_tp2`**: AWQ is only 10% faster and changes the ranking (MAE 0.20) |
| 3 judge (2×T4) | `bn-full-s03-judge-01`, `-02`, ... | **S01 DONE** (623 min); **S02 DONE** (629 min, 90,000 / 91,120); **S03 DONE** (24 min) → **Stage 3 COMPLETE: 91,120 / 91,120, parse 1.00** | S01: **46,000 / 91,120** scored, 4,458 rows/h, parse 1.00, clean stop at the time budget, no crash or eager retry. Startup 9 min, 26.7 min per chunk. S02: +44,000 -> **90,000 / 91,120**, 4,244 rows/h (28 min per chunk, 5% slower than S01), parse 1.00, stopped at the budget **1,120 rows short**. Merged parquet checked: 90,000 unique ids, no nulls, S01 scores identical; S02's new rows have the same c/q distributions. S03 needs ~15 min scoring + ~9 min startup. On the S01 rows, c: median 4.00, p10 2.18, p90 4.79. **q has a ceiling**: median 5.00, 71% ≥ 4.9, but 18% < 4 and 7% < 3. Spearman(c, q) = −0.23. Median c by domain: translation 2.23, mcq 2.78, other 2.95, the rest 3.7–4.0 |
| 4 select arms (CPU) | `bn-full-s04-select` | **DONE: v4** (τ 0.74, arm F from the Qwen IFD) | v1 τ 0.90, v2 sweep, v3 τ 0.74 + Llama IFD, **v4 = final**. Arms A 91,120, B–F 10,000; held-out 470; in `outputs/full/s04_v4/` (`s04/` = v3, `s04_v1_tau090/` = v1, both stale). A–E identical to v3. τ 0.74 ≈ pool NN-sim p25 (0.746): **C∩E 58%**. C / E / B: mean_pair_sim .379 / .391 / .386, distinct-2 .503 / .451 / .515, coverage .746 / .727 / .757, c_mean 4.49 / 4.74 / 3.71. Code share: pool 11.4%, C 3.0%, E 1.4%, **F 21.2%** (v3 Llama F: 7.9%). Tokens: B 12.2M, C 15.5M, D 15.5M, E 15.2M, **F 10.9M**. C∩F 7.3%. D cut q ≥ 5.44 |
| 5 LoRA per arm (2×T4) | `habiburrahman118/bn-full-s05-lora-NN` | **sessions 01, 02 DONE** (419, 318 min): 9 runs complete; **session 03 RUNNING** (arm A part 1 + D43, E43, F43, F44; pushed 2026-10-30 20:29 BST, expect done ~07:30 BST 2026-10-31) | `kaggle/full_run/s05_lora/`. Qwen2.5-0.5B-Instruct, pilot hyper-parameters, per-GPU queues, checkpoint every 250 steps, stop on a time budget, resume by attaching the previous session. S01: B, C, D, E, F seed 42, all 2,500 / 2,500 steps, no NaN, peak 7.6 GB; training min B 131, C 171, D 176, E 162, F 116; output in `outputs/full/s05_01/`. S02: seeds 43, 44 for B and C, all complete (B 135, C 174 min); carried-over seed 42 files byte-identical to S01; output in `outputs/full/s05_02/` |
| 6 eval (CPU, local) | `kaggle/full_run/s06_eval/s06_bootstrap.py` | **DONE for S02** (B, C 3 seeds; D, E, F 1 seed); rerun after each new session | `results/full_run/`. 470 held-out rows. Per-example mean loss (paired diff vs B, 95% CI): base .978, **B .813**, F .827 (+.014 [.011, .017]), C .838 (+.025 [.019, .032]), D .857 (+.044 [.035, .054]), E .863 (+.050 [.039, .062]). Token-weighted: B .8172, C .8186 (+.0014 [−.0004, .0033], tie on the CI, but all 3 C seeds .8185–.8188 are above all 3 B seeds .8170–.8176), E .825, D .831, F .833. **Seed spread is small**: B .8120–.8130, C .8377–.8385 (per example), so the C−B gap of .025 is ~25× the seed spread. **Code drives it** (70 rows): F .843, B .861, C .995, D 1.027, E 1.124; 78% of the C−B gap. Non-code (400 rows): B .804, C .811, E .817, F .824, D .827. No arm beats random |

Stage 1 caveat: the filter-bias check flagged **code (20.7% dropped, mostly `language`: 3,038 rows)** and
**mcq_classification (21.0%, mostly `translationese`: 366 rows)**. The pilot showed the same pattern (code 19%).
Likely cause: code outside ``` fences counts as English prose. The held-out set is filtered the same way, so the
comparison between arms stays fair, but say so in the thesis, or loosen the rule and rerun Stage 1 before Stage 3.

## Next action

Account `habiburrahman153` has no GPU quota left after Stage 2b (it had 2.5 h on 2026-10-30; the quota resets weekly).
CPU kernels (Stage 4) do not use GPU quota and can still run there. **All GPU work from Stage 5 on runs on a second account.**

1. ~~Stage 2b~~ done (91,120 rows, 0 NaN). ~~Stage 4 v4~~ done (arm F rebuilt from the Qwen IFD; A–E unchanged).
   Arms downloaded and verified (A 91,120, B–F 10,000, held-out 470; unique ids, no empty rows, held-out disjoint from the pool) in `outputs/full/s04_v4/bangla_filtering/full/arms/`. **Account habiburrahman153 is finished for now** (~0.8 h GPU left).
2. ~~Move to the second account~~ done: CLI token = **habiburrahman118** (account 1's token kept as
   `~/.kaggle/access_token.habiburrahman153`). Arms uploaded as the private dataset **`habiburrahman118/bn-full-arms`**
   (A–F, held-out, `stage4_manifest.json`, `arm_diagnostics.csv`). Note: `kaggle datasets create -p <relative path>` fails on
   Windows (temp-path bug); run it from inside the folder with `-p .`.
3. ~~Stage 5 sessions 01, 02~~ done (9 runs complete); ~~Stage 6 on them~~ done (`results/full_run/`).
   **Stage 5 session 03 is RUNNING** (`habiburrahman118/bn-full-s05-lora-03`, pushed 2026-10-30 20:29 BST; session 02 attached; budget 11 h).
   GPU0: arm A seed 42 (22,780 steps, ~20.6 h); GPU1: D43 → E43 → F43 → F44 (~9.8 h). Expect done ~07:30 BST 2026-10-31. Check it, then rerun Stage 6:
   ```bash
   export PYTHONUTF8=1
   kaggle kernels status habiburrahman118/bn-full-s05-lora-03
   kaggle kernels output habiburrahman118/bn-full-s05-lora-03 -p outputs/full/s05_03
   cat outputs/full/s05_03/bangla_filtering/full/lora/stage5_summary.json
   python kaggle/full_run/s06_eval/s06_bootstrap.py outputs/full/s05_03
   ```
   Expect 13 runs `complete: true` and `arm_A_seed42` `complete: false` at about step 11,700, with a `checkpoint-*` folder; the log ends with INCOMPLETE for arm A, which is normal.
4. **Stage 5 session 04**: in `s05_lora.py` set `QUEUES = {0: [("A", 42)], 1: [("D", 44), ("E", 44)]}`; in `kernel-metadata.json` set the id to `-04` and `kernel_sources` to `-03`.
   **Check the quota first**: ~6.4 h left this week after S03, and A needs ~10 h more. Wait for the weekly reset, or set `TIME_BUDGET_H` under the quota left and run a session 05. Do not let a session outrun the quota: a killed session loses its output.
   Then rerun Stage 6, then **decide the downstream benchmarks** (ROADMAP phase 4).
5. Report in the paper: token totals differ per arm (C +27%, F −10% vs B), so compare per token as well as per row; code is under-selected by c × q.

## Decisions

- Arm size **K = 10,000** for arms B–F (user, 2026-10-26). Set `SUBSET_SIZE = 10_000` in Stage 4.
- **τ = 0.74** (2026-10-30): the notebook's 0.90 left C 91.5% identical to E; 0.74 ≈ pool NN-sim p25, C∩E 58%.
- **Base model for LoRA: Qwen2.5-0.5B-Instruct** (user, 2026-10-30): ~2.4 h per 10K arm vs ~5–6 h for Llama-3.2-1B; IFD recomputed with it (Stage 2b) so arm F matches.
- **Kaggle accounts** (2026-10-30): stages 1–4 on `habiburrahman153`; Stage 5 onward on **`habiburrahman118`**.
- Arm A (whole pool, 91,120 rows): **full run** (2026-10-30). The user said to run the next stage and did not choose explicitly, so Claude took the standing recommendation; the user can still overrule it. A compute-matched run (2,500 steps on the whole pool) is just another random 10K rows, i.e. arm B, so it cannot answer "how close to the whole pool". 102.4M tokens at ~12 min/M = ~20.6 h, so 2 sessions at an 11 h budget.
- Free GPU during arm A: extra seeds for D, E, F (43, 44), so all arms end with 3 seeds; quota counts session time, so these cost nothing extra.
- `habiburrahman118` GPU quota: ~12.6 h used this week before S03 (S01 7 h + S02 5.3 h + smoke + resume test), ~23.6 h after it, so ~6.4 h left of ~30 h; arm A's second session needs the next week's quota or a shorter budget.
- Resumed runs: `train_loss` in `training_report.json` and `stage5_summary.json` is too low (resume test 0.53 vs 0.94), because it counts only the current session's loss. The log around the stop step is also a partial average. Weights and held-out losses are unaffected. Do not quote it for arm A; use the mean of the `loss` entries in `log_history`.

## Conventions

- Accounts: `habiburrahman153` (stages 1–4, GPU quota used up 2026-10-30) and `habiburrahman118` (Stage 5+). Kernel ids in
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
- 2026-10-30: Stage 2b done (101 min): Qwen IFD for all 91,120 rows, 0 NaN, Spearman 0.68 vs Llama IFD. Stage 4 v4 done: arm F rebuilt (code 21.2%, 10.9M tokens), A–E unchanged. Arms downloaded. Account 1 done; next is the account switch.
- 2026-10-30: arms verified and uploaded as private dataset `habiburrahman118/bn-full-arms` (CLI switched to habiburrahman118). Wrote Stage 5 (`s05_lora.py`: per-GPU queues, checkpoint/resume, time budget, per-example held-out losses).
- 2026-10-30: dataset upload done. Stage 5 smoke test passed (6.7 min, all 5 runs, both GPUs). Moved the adapter save before the held-out eval. **Pushed Stage 5 session 01** (02:46 BST).
- 2026-10-30: Stage 5 session 01 done (419 min): B–F seed 42 complete, files verified (470 ids, 0 NaN). Stage 6 on seed 42: B best on per-example loss, C ties B token-weighted, code explains 80% of C−B. `compare_arms_bootstrap.py` got an optional OUT_DIR so the pilot's results stay untouched. Pushed Stage 5 session 02 (13:51 BST): seeds 43, 44 for B and C, one of each per GPU (~5.2 h, vs ~5.8 h with both C runs on one GPU).
- 2026-10-30: Stage 5 session 02 done (318 min): B, C seeds 43, 44 complete. Stage 6 rerun on 9 runs: seed spread ~.001, far below the arm gaps; B still best. Arm A would be ~20.6 h (~102M Qwen tokens at ~12 min/M) and needs the resume path, never exercised so far, so pushed a short test kernel `bn-full-s05-resume-test` (19:50 BST) generated from the real `s05_lora.py`.
- 2026-10-30: resume test done (12 min): a 40-step run stopped at step 17 and resumed from a copied checkpoint matches an uninterrupted run exactly (max weight diff 0.0, per-example loss diff 0.0; a repeat of the uninterrupted run is also 0.0). Only `train_loss` is wrong after a resume (see Decisions). Full arm A tokenizes in 5.0 min at 14.6 GB peak RSS: 91,118 of 91,120 rows used, 102,413,501 tokens, 238 truncated, 2 dropped. Pushed Stage 5 session 03 (20:29 BST): arm A seed 42 on GPU 0, D43, E43, F43, F44 on GPU 1, budget 11 h.
