# Full-run plan: 100K-row pool on Kaggle

Scale the Iteration 03 pilot (1,000 rows) to the full 100K pool. We have no local GPU, so every stage
runs as a private Kaggle **script kernel**. Each stage reads the previous stage's output by attaching
that kernel as an input (`kernel_sources`).

Data: `md-nishat-008/Bangla-Instruct` has **342,391 rows**. As in the pilot notebook, we draw a
domain-stratified sample of 100,000 pool rows + 500 held-out rows (seed 42). The held-out rows are
set aside before any filtering and never trained on.

Live status and the next command to run: [HANDOFF.md](../../HANDOFF.md).

## Kaggle limits we plan around

- GPU quota: about 30 h per week. A session runs at most 12 h; we stop scoring at 10.5 h and save partial results.
- `machine_shape: NvidiaTeslaT4` gives **2 × T4** (16 GB each, no native bf16). Quota counts session
  time, not GPUs, so every GPU stage keeps both T4s busy.
- CPU kernels do not use GPU quota.
- Account `habiburrahman153` used up its weekly GPU quota on 2026-10-30 (stages 1–3 + 2b). Stage 5 onward runs on
  `habiburrahman118`; a private kernel's output cannot be attached across accounts, so the arms move over as a private dataset (HANDOFF.md).
- A kernel whose session is killed at 12 h loses its output, so scoring stages checkpoint and stop on a time budget.

## Stages

| # | Kernel | Hardware | Does | Est. time | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | `bn-full-s01-clean` | CPU | sample 100K + 500 held-out, normalise, filters, dedup, decontamination, token counts | 11 min | done: 91,120 pool / 470 held-out |
| 2 | `bn-full-s02-ifd-embed` | 2×T4 | IFD with Llama-3.2-1B (one half of the rows per GPU), bge-m3 embeddings, τ calibration stats | ~3.5 h | done: 223 min |
| 2b | `bn-full-s02b-ifd-qwen` | 2×T4 | IFD again with Qwen2.5-0.5B-Instruct, the model Stage 5 trains (arm F) | ~1.5 h | done: 101 min, 0 NaN, Spearman 0.68 vs Llama IFD |
| 3a | `bn-full-s03a-judge-bench` | 2×T4 | judge throughput benchmark on the 955 pilot rows: HF backend vs vLLM fp16 TP2, AWQ and GPTQ-Int8 (one copy per GPU), Spearman agreement with the pilot scores | ~1 h | done: vllm_tp2 chosen (4,374 rows/h, ρ 1.000) |
| 3 | `bn-full-s03-judge-NN` | 2×T4 | LLM judge c, q for all rows; each run resumes from the previous runs' outputs | ~21 h (2–3 sessions) | done: 91,120 rows in 3 sessions (S01 623 min, S02 629, S03 24), parse 1.00 |
| 4 | `bn-full-s04-select` | CPU | build arms A–F, recalibrate τ, diagnostics | < 1 h | done: v4 (τ 0.74, C∩E 58%; arm F from the Qwen IFD) |
| 5 | `habiburrahman118/bn-full-s05-lora-NN` | 2×T4 | LoRA (Qwen2.5-0.5B-Instruct), a queue of runs per GPU; arms from the private dataset `habiburrahman118/bn-full-arms`; per-example held-out losses | 1.9–2.9 h per 10K-row arm; B–F in one ~7 h session | S01 done (B–F seed 42, 419 min); S02 running (seeds 43, 44 for B, C) |
| 6 | `kaggle/full_run/s06_eval/s06_bootstrap.py` | CPU, local | paired bootstrap on the held-out losses Stage 5 wrote (470 rows), per-example and token-weighted | seconds | done for seed 42; rerun after S02 |

Estimated GPU total: about 60–90 h, i.e. 2–3 weeks of Kaggle quota. The judge is the largest cost.

## The judge (Stage 3) is the bottleneck

Pilot: 3,216 s for 955 rows, but the pilot scored 4 prompts per row (English + Bengali rubric × complexity, quality).
Full mode uses the English rubric only, so 2 prompts per row: **about 47 GPU-hours** with the HF backend.

On a T4, Qwen2.5-7B-Instruct in fp16 (15 GB) does not fit on one GPU, so the HF backend splits it across both.
Only one GPU computes at a time. Options that Stage 3a measures:

1. **HF, as in the pilot**: no change in scores, about 47 h.
2. **vLLM, tensor-parallel over 2 T4s**: same weights, so scores should match. Expect 2–4× faster if vLLM runs on T4 (compute capability 7.5) with the Kaggle image.
3. **AWQ 4-bit 7B, one copy per GPU** (`Qwen/Qwen2.5-7B-Instruct-AWQ`): fastest, but the scores change slightly. Accept only if Spearman with the pilot scores is ≥ 0.9.

## Decisions still open (recommendations in bold)

- ~~Arm size K~~ **DECIDED 2026-10-26: K = 10,000** (10% of the pool, about 2.5 h of T4 time per arm).
  The notebook default of 40,000 would take about 10 h per arm.
- **Arm A (whole pool, ~95K rows)**: about 24 h of training on its own. **Train it last, with checkpoint resume over 2–3 sessions**,
  or replace it with a compute-matched run (the same number of steps as a 10K arm).
- ~~τ (diversity threshold)~~ **DECIDED 2026-10-30: τ = 0.74** ≈ the pool's NN-sim p25 (0.746); C∩E 58% (0.90 gave 91.5%).
- ~~Base model~~ **DECIDED 2026-10-30: Qwen2.5-0.5B-Instruct** (as in the pilot; IFD recomputed with it in Stage 2b).
- **Seeds**: train B and C (the main comparison) with 3 seeds once single-seed results are in.
