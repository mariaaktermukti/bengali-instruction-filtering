# HANDOFF: full 100K run on Kaggle

Read this first in a new session. Plan and reasoning: [kaggle/full_run/PLAN.md](kaggle/full_run/PLAN.md).
Update the **Status** table and **Next action** after every step, and add a line to the log.

## Status

| Stage | Kernel (habiburrahman153/...) | State | Output / key numbers |
| --- | --- | --- | --- |
| 1 clean (CPU) | `bn-full-s01-clean` | **DONE** (v1, ~11 min) | 100K sampled -> **91,120 clean pool**, **470 held-out**. Drops: heuristics 5,081 (language 4,215), exact dups 2,991, near dups 736, decontamination 71 (62 held-out, 8 MMLU-bn). 111.5M Llama tokens |
| 2 IFD + embeddings (2×T4) | `bn-full-s02-ifd-embed` | **DONE** (v1, 223 min: IFD 192, embed 30.5) | IFD for all 91,120 rows (IFD ≥ 1: 0.02%). Median IFD by domain 0.75 (code) to 0.88. bge-m3 sanity: random median 0.38, p99 0.58; MinHash near-dups median 0.98, 100% above random p99. **NN-sim quantiles** p10 .680, p25 .746, p50 .815, p75 .883, p90 .939, p95 .959, p99 .978 |
| 3a judge benchmark (2×T4) | `bn-full-s03a-judge-bench` | **RUNNING** (v1 pushed 2026-10-27, about 1 h expected; not tested before this run) | - |
| 3 judge | - | not written | - |
| 4 select arms | - | not written | - |
| 5 LoRA per arm | - | not written | - |
| 6 eval | - | not written | - |

Stage 1 caveat: the filter-bias check flagged **code (20.7% dropped, mostly `language`: 3,038 rows)** and
**mcq_classification (21.0%, mostly `translationese`: 366 rows)**. The pilot showed the same pattern (code 19%).
Likely cause: code outside ``` fences counts as English prose. The held-out set is filtered the same way, so the
comparison between arms stays fair, but say so in the thesis, or loosen the rule and rerun Stage 1 before Stage 3.

## Next action

1. Check Stage 3a:
   ```bash
   export PYTHONUTF8=1
   kaggle kernels status habiburrahman153/bn-full-s03a-judge-bench
   kaggle kernels output habiburrahman153/bn-full-s03a-judge-bench -p outputs/full/s03a --file-pattern "json|txt|\.log"
   ```
   Read `outputs/full/s03a/bangla_filtering/full/judge_bench/judge_bench.json`. For each backend it gives `rows_per_h`,
   `est_hours_full_pool` (91,120 rows), `spearman_{c,q}_vs_pilot`, `parse_rate`, `max_prompt_tokens`, and `recommended`
   (the fastest with both Spearman ≥ 0.9 and parse rate ≥ 0.95). `hf` vs pilot is the noise floor: the same code reran.
   If a vLLM backend failed, `error_tail` and `log_<backend>_s<k>_try<n>.txt` show why (each was retried once with `enforce_eager`).
   Backends: `hf` (pilot code, 300 rows), `vllm_tp2` (fp16, tensor-parallel 2), `awq_dp2`, `gptq8_dp2` (one copy per GPU), all 955 pilot rows.
2. Write Stage 3 (full judge, `kaggle/full_run/s03_judge/`) with the recommended backend: English rubric only, input = Stage 1
   `pool_clean.parquet`, checkpoint every chunk, stop at 10.5 h, resume from earlier runs' outputs attached as `kernel_sources`
   (same pattern as `find_previous` in Stage 2). Reuse the worker code from `s03a_judge_bench.py`; for a `_dp2` backend, split rows by `iloc[k::2]`.
3. Stage 4 τ: the notebook default τ = 0.90 sits between the pool's NN-sim p75 (.883) and p90 (.939), so about 20% of rows have a
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
