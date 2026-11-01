# Roadmap to the paper

One line per step. Tick a box when the step is done and add the date. Run-level detail (commands, numbers)
lives in [HANDOFF.md](HANDOFF.md); the stage design lives in [kaggle/full_run/PLAN.md](kaggle/full_run/PLAN.md).

**Research question:** for Bengali instruction tuning, does selecting a 10K subset by LLM-judged complexity × quality
(+ a diversity step, DEITA-style) beat random selection, top-quality-only, or IFD (Cherry-LLM), and how close does it get
to training on the whole clean pool?

## Progress

| Phase | Done | Total |
| --- | --- | --- |
| 0. Pilot (1K rows) | 4 | 5 |
| 1. Full-run data pipeline | 5 | 5 |
| 2. Arm selection | 4 | 5 |
| 3. Training | 6 | 6 |
| 4. Evaluation | 2 | 5 |
| 5. Analysis | 0 | 5 |
| 6. Writing | 0 | 8 |
| **All** | **21** | **39** |

Status as of 2026-10-31: the expensive data and training work is finished. **Arm A (the full 91,120-row pool) is best overall at .7169 held-out loss; among 10K arms, random B is best at .8126.** C is +.0254 vs B per-example, but only +.0014 token-weighted with a CI crossing zero. F is +.0133 vs B per-example. The seed spread remains small (~.001).

The Stage 6 bootstrap has completed on Kaggle CPU. No further GPU run is planned until downstream benchmarks are chosen.
**Data and arms are final** (Stage 4 v4). **Update 2026-10-31:** Stage 5 session 04 completed: A42 reached 22,780 / 22,780 (held-out loss .7169), D44/E44 completed, and all 16 runs are valid. Stage 6 CPU kernel `habiburrahman118/bn-full-s06-eval-01` completed on Kaggle against S04; refreshed results are in `results/full_run/`. Details: [HANDOFF.md](HANDOFF.md) → Next action.

## 0. Pilot (Iteration 03, 1,000 rows)

- [x] Filtering + selection notebook on 1K rows (`kaggle/notebooks/bangla-filtering-selection.ipynb`)
- [x] LoRA arms A, B, C, D, F on the pilot (Qwen2.5-0.5B-Instruct), held-out eval + paired bootstrap ([KAGGLE_EXPERIMENTS.md](KAGGLE_EXPERIMENTS.md)). Finding: no 400-row rule beats random; IFD is worse
- [x] Judge backend benchmark (Stage 3a): vLLM fp16 TP2, Spearman 1.000 vs the pilot (2026-10-27)
- [x] Judge EN vs BN rubric agreement on the pilot (notebook cell 23)
- [ ] Judge vs **human** labels (100 rows: `scripts/annotate_human.py`, `analysis/validate_llm_judge.py`). Labels are now at `human_annotation_100.jsonl`, but only 29 of the 100 rows are in the full-run pool, so the judge must score these rows (HANDOFF.md). A reviewer will ask for this

## 1. Full-run data pipeline (100K pool)

- [x] Stage 1 clean: 100K sampled → 91,120 pool, 470 held-out (2026-10-26)
- [x] Stage 2 IFD (Llama-3.2-1B) + bge-m3 embeddings (2026-10-27)
- [x] Stage 3 LLM judge c, q for all 91,120 rows, 3 sessions, parse 1.00 (2026-10-29)
- [x] Score sanity: q has a ceiling (71% ≥ 4.9), Spearman(c, q) = −0.24, Spearman(c, instr length) = 0.47 (2026-10-29)
- [x] Stage 2b: IFD recomputed with Qwen2.5-0.5B-Instruct (the model we train), 91,120 rows, 0 NaN, Spearman 0.68 vs Llama IFD (2026-10-30)

## 2. Arm selection (Stage 4, CPU)

- [x] Stage 4 v1, τ = 0.90: arms A (91,120), B–F (10,000 each) (2026-10-29)
- [x] Diagnostics: C∩E = 91.5% at τ 0.90 (diversity step barely acts); code share 11.4% → 1.6% in C/E; token totals differ (B 12.2M, C 15.3M)
- [x] τ recalibration: sweep 0.65–0.96, chose **τ 0.74** (≈ pool NN-sim p25; C∩E 58%, C has the lowest pair-sim .379 and code share 3.0% vs E 1.4%); arms rebuilt (Stage 4 v3) (2026-10-30)
- [x] Stage 4 v4: arm F rebuilt with the Qwen IFD (code share 21.2%, 10.9M tokens); A–E unchanged (2026-10-30)
- [ ] Blind spot-check: rate `reports/spotcheck_blind.csv` (50 C + 50 B rows, arm hidden), then join with `spotcheck_key.csv`

## 3. Training (Stage 5, 2×T4, two arms per session)

- [x] Base model: **Qwen2.5-0.5B-Instruct** (~2.4 h per 10K arm vs ~5–6 h for Llama-3.2-1B; matches the pilot). IFD redone with it in Stage 2b (2026-10-30)
- [x] Switched to Kaggle account **habiburrahman118**; arms uploaded as private dataset `habiburrahman118/bn-full-arms` (2026-10-30)
- [x] Stage 5 kernel (`kaggle/full_run/s05_lora/`: per-GPU queues, checkpoint/resume), smoke test passed (2026-10-30)
- [x] Train B, C, D, E, F (seed 42) in one session, all complete, 419 min (2026-10-30)
- [x] Extra seeds for B and C (2 more each), for confidence intervals on the main comparison; seed spread ~.001 (2026-10-30)
- [x] Arm A (whole pool), full run (~20.6 h); resume path tested, exact match ← *completed in S04 at 22,780 / 22,780 steps (2026-10-31)*. D43, E43, F43, D44, E44 complete; B–F now have 3 seeds

## 4. Evaluation (Stage 6)

- [x] Held-out loss per model on the 470 held-out rows, written by Stage 5 (2026-10-30)
- [x] Paired bootstrap vs B and vs base, by domain, token-weighted (`kaggle/full_run/s06_eval/s06_kaggle.py`, `results/full_run/`); Kaggle CPU run `bn-full-s06-eval-01` completed against all 16 S04 runs (2026-10-31)
- [ ] **Decide on downstream benchmarks.** Held-out loss alone is weak evidence for a paper. Recommended, waiting for the user's go: Bengali MMLU, BanglaRQA, and blind pairwise judging of generations on the held-out prompts. mHumanEval-bn is left out because it is a safety benchmark, not a test of instruction quality
- [ ] Run the downstream benchmarks for every model ← *Stage 7 kernel ready and CPU-tested, not pushed (`kaggle/full_run/s07_benchmarks/`); pairwise judging is Stage 8*
- [ ] Generation samples per arm (qualitative examples for the paper)

## 5. Analysis

- [ ] Main results table: arm × (held-out loss, benchmarks), with CIs
- [ ] Per-token view: results vs training tokens (arms differ by up to 26% in tokens)
- [ ] Domain-mix analysis: does under-selecting code hurt code-related eval?
- [ ] τ / diversity ablation: C vs E at the chosen τ (and the sweep table)
- [ ] Filter-bias appendix: Stage 1 drops 20.7% of code rows (code outside fences counts as English)

## 6. Writing

- [ ] Outline + target venue and page limit
- [ ] Introduction and related work (DEITA, Cherry-LLM/IFD, TigerLLM, Bangla-Instruct)
- [ ] Data and pipeline section (filters, dedup, decontamination, judge, IFD, embeddings)
- [ ] Selection methods section (arms A–F, τ calibration)
- [ ] Experimental setup (model, LoRA config, compute)
- [ ] Results and analysis
- [ ] Limitations (judge q ceiling, synthetic GPT-4o/Claude source data, single base model, filter bias on code)
- [ ] Abstract, conclusion, final pass

## Decisions

| Decision | Status |
| --- | --- |
| Arm size K | **10,000** (2026-10-26) |
| Judge backend | **vLLM fp16 TP2** (2026-10-27) |
| τ | **0.74** (2026-10-30) |
| Base model for LoRA | **Qwen2.5-0.5B-Instruct** (2026-10-30) |
| Kaggle account | stages 1–4: habiburrahman153; Stage 5+: **habiburrahman118** |
| Arm A: full or compute-matched | **full** (2026-10-30; Claude took the standing recommendation, user can overrule) |
| Downstream benchmarks | recommended: MMLU-bn, BanglaRQA, pairwise judging; waiting for the user's go |
| Pairwise judge (Stage 8) | open; recommended: a judge not used in selection, plus a human check on ~100 pairs |
