# HANDOFF: full 100K run on Kaggle

Read this first in a new session. Plan and reasoning: [kaggle/full_run/PLAN.md](kaggle/full_run/PLAN.md).
Update the **Status** table and **Next action** after every step, and add a line to the log.

## Status

| Stage | Kernel (habiburrahman153/...) | State | Output / key numbers |
| --- | --- | --- | --- |
| 1 clean (CPU) | `bn-full-s01-clean` | **DONE** (v1, ~11 min) | 100K sampled -> **91,120 clean pool**, **470 held-out**. Drops: heuristics 5,081 (language 4,215), exact dups 2,991, near dups 736, decontamination 71 (62 held-out, 8 MMLU-bn). 111.5M Llama tokens |
| 2 IFD + embeddings (2×T4) | `bn-full-s02-ifd-embed` | **DONE** (v1, 223 min: IFD 192, embed 30.5) | IFD for all 91,120 rows (IFD ≥ 1: 0.02%). Median IFD by domain 0.75 (code) to 0.88. bge-m3 sanity: random median 0.38, p99 0.58; MinHash near-dups median 0.98, 100% above random p99. **NN-sim quantiles** p10 .680, p25 .746, p50 .815, p75 .883, p90 .939, p95 .959, p99 .978 |
| 3a judge benchmark (2×T4) | `bn-full-s03a-judge-bench` | **DONE** (v2, 62 min; v1 was partial) | vLLM 0.32.0 works after uninstalling torchaudio (torch 2.13+cu130, cuda True, 2 GPUs). Rows/h and hours for 91,120 rows, Spearman c vs pilot / MAE: `hf` 2,763 (33.0 h, 0.9999 / 0.003); **`vllm_tp2` 4,374 (20.8 h, 1.0000 / 0.0027)**; `awq_dp2` 4,814 (18.9 h, 0.962 / 0.20, q 0.923); `gptq8_dp2` 4,243 (21.5 h, 0.999 / 0.014). Parse 1.00 everywhere, max prompt 3,131 tok. Script `recommended` = `awq_dp2`, but the rule is max rows/h with no quality check. **Use `vllm_tp2`**: AWQ is only 10% faster and changes the ranking (MAE 0.20) |
| 3 judge (2×T4) | `bn-full-s03-judge-01`, `-02`, ... | **S01 DONE** (623 min); **S02 DONE** (629 min, 90,000 / 91,120); **S03 RUNNING** (pushed 2026-10-29, 1,120 rows left) | S01: **46,000 / 91,120** scored, 4,458 rows/h, parse 1.00, clean stop at the time budget, no crash or eager retry. Startup 9 min, 26.7 min per chunk. S02: +44,000 -> **90,000 / 91,120**, 4,244 rows/h (28 min per chunk, 5% slower than S01), parse 1.00, stopped at the budget **1,120 rows short**. Merged parquet checked: 90,000 unique ids, no nulls, S01 scores identical; S02's new rows have the same c/q distributions. S03 needs ~15 min scoring + ~9 min startup. On the S01 rows, c: median 4.00, p10 2.18, p90 4.79. **q has a ceiling**: median 5.00, 71% ≥ 4.9, but 18% < 4 and 7% < 3. Spearman(c, q) = −0.23. Median c by domain: translation 2.23, mcq 2.78, other 2.95, the rest 3.7–4.0 |
| 4 select arms (CPU) | `bn-full-s04-select` | **written + tested locally, not pushed** (waits for S03) | `kaggle/full_run/s04_select/`. Ports notebook cells 22–31 (numpy DEITA, no GPU). Adds a τ sweep (0.80–0.96: depth, rejected rows, C∩E overlap) in `reports/tau_sweep.csv`; arms use `TAU_SIM` env (default 0.90). Refuses to run if judge coverage < 99% or parse rate < 0.95. Synthetic test: greedy matches a naive reference exactly, max pair-sim in C < τ |
| 5 LoRA per arm | - | not written | - |
| 6 eval | - | not written | - |

Stage 1 caveat: the filter-bias check flagged **code (20.7% dropped, mostly `language`: 3,038 rows)** and
**mcq_classification (21.0%, mostly `translationese`: 366 rows)**. The pilot showed the same pattern (code 19%).
Likely cause: code outside ``` fences counts as English prose. The held-out set is filtered the same way, so the
comparison between arms stays fair, but say so in the thesis, or loosen the rule and rerun Stage 1 before Stage 3.

## Next action

1. Stage 3 runs one kernel per session, because a kernel cannot attach its own output. Each session attaches
   `bn-full-s01-clean` plus **all** earlier judge kernels; `find_previous` merges their `judge_en.parquet` files and skips rows already scored.
   Sessions 01 and 02 are done (90,000 rows; outputs in `outputs/full/s03_01/`, `outputs/full/s03_02/`). S02 stopped 1,120 rows short.
   `kernel-metadata.json` now points at **session 03** (id `bn-full-s03-judge-03`, attaches s01-clean + judge-01 + judge-02). **Pushed 2026-10-29** (v1). Check ~30 min later:
   ```bash
   export PYTHONUTF8=1
   kaggle kernels push -p kaggle/full_run/s03_judge
   kaggle kernels status habiburrahman153/bn-full-s03-judge-03
   kaggle kernels output habiburrahman153/bn-full-s03-judge-03 -p outputs/full/s03_03 --file-pattern "json|\.log|parquet"
   ```
   Expect `complete: true` and `scored` 91,120.
   The final scores are the `judge_en.parquet` of the last session; it holds all rows (columns `id`, `complexity_en`,
   `complexity_en_mass`, `quality_en`, `quality_en_mass`; Stage 4 sets `parse_ok`, `c` and `q` as in cell 22 of the notebook).
   Then push Stage 4 (CPU, no GPU quota; attaches s01, s02, judge-01/02/03):
   ```bash
   kaggle kernels push -p kaggle/full_run/s04_select
   kaggle kernels status habiburrahman153/bn-full-s04-select
   kaggle kernels output habiburrahman153/bn-full-s04-select -p outputs/full/s04 --file-pattern "json|csv|\.log"
   ```
   (the arm jsonl files are large; download them only when Stage 5 needs them locally, Stage 5 attaches the kernel instead).
2. Stage 4 τ: the notebook default τ = 0.90 sits between the pool's NN-sim p75 (.883) and p90 (.939), so about 20% of rows have a
   neighbour above 0.90. Inside a 10K selection the neighbours are sparser, so fewer rows are rejected. Check how many rows the
   diversity step rejects in Stage 4 and adjust τ so that arms C and E differ.

## Decisions

- Arm size **K = 10,000** for arms B–F (user, 2026-10-26). Set `SUBSET_SIZE = 10_000` in Stage 4.
- Arm A (whole pool, ~95K rows, ~24 h): still open. Options: multi-session training with resume, or a compute-matched run.

## Conventions

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
