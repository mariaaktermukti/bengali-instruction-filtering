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
| 1. Full-run data pipeline | 4 | 5 |
| 2. Arm selection | 3 | 5 |
| 3. Training | 1 | 7 |
| 4. Evaluation | 0 | 5 |
| 5. Analysis | 0 | 5 |
| 6. Writing | 0 | 8 |
| **All** | **12** | **40** |

Status as of 2026-10-30: the expensive data work is finished (~45 GPU-hours spent). **No full-scale model results exist yet**,
and the paper's results section depends entirely on phases 3–5.

Estimated GPU time left: about 20–45 h (1–2 weeks of the Kaggle quota, ~30 h per week), mostly LoRA training.
**Now:** Stage 2b (IFD with Qwen) is running on account habiburrahman153, its last GPU job; then Stage 4 v4 (CPU) rebuilds arm F.
**Next:** move to a second Kaggle account (GPU quota on the first is used up) and write Stage 5. Details: [HANDOFF.md](HANDOFF.md) → Next action.

## 0. Pilot (Iteration 03, 1,000 rows)

- [x] Filtering + selection notebook on 1K rows (`kaggle/notebooks/bangla-filtering-selection.ipynb`)
- [x] LoRA arms A, B, C, D, F on the pilot (Qwen2.5-0.5B-Instruct), held-out eval + paired bootstrap ([KAGGLE_EXPERIMENTS.md](KAGGLE_EXPERIMENTS.md)). Finding: no 400-row rule beats random; IFD is worse
- [x] Judge backend benchmark (Stage 3a): vLLM fp16 TP2, Spearman 1.000 vs the pilot (2026-10-27)
- [x] Judge EN vs BN rubric agreement on the pilot (notebook cell 23)
- [ ] Judge vs **human** labels (100 rows: `scripts/annotate_human.py`, `analysis/validate_llm_judge.py`). Results are not in the repo; find or redo them. A reviewer will ask for this

## 1. Full-run data pipeline (100K pool)

- [x] Stage 1 clean: 100K sampled → 91,120 pool, 470 held-out (2026-10-26)
- [x] Stage 2 IFD (Llama-3.2-1B) + bge-m3 embeddings (2026-10-27)
- [x] Stage 3 LLM judge c, q for all 91,120 rows, 3 sessions, parse 1.00 (2026-10-29)
- [x] Score sanity: q has a ceiling (71% ≥ 4.9), Spearman(c, q) = −0.24, Spearman(c, instr length) = 0.47 (2026-10-29)
- [ ] Stage 2b: IFD recomputed with Qwen2.5-0.5B-Instruct (the model we train), so arm F follows Cherry-LLM ← *running, pushed 2026-10-30*

## 2. Arm selection (Stage 4, CPU)

- [x] Stage 4 v1, τ = 0.90: arms A (91,120), B–F (10,000 each) (2026-10-29)
- [x] Diagnostics: C∩E = 91.5% at τ 0.90 (diversity step barely acts); code share 11.4% → 1.6% in C/E; token totals differ (B 12.2M, C 15.3M)
- [x] τ recalibration: sweep 0.65–0.96, chose **τ 0.74** (≈ pool NN-sim p25; C∩E 58%, C has the lowest pair-sim .379 and code share 3.0% vs E 1.4%); arms rebuilt (Stage 4 v3) (2026-10-30)
- [ ] Stage 4 v4: rebuild arm F with the Qwen IFD (CPU; A–E stay as in v3)
- [ ] Blind spot-check: rate `reports/spotcheck_blind.csv` (50 C + 50 B rows, arm hidden), then join with `spotcheck_key.csv`

## 3. Training (Stage 5, 2×T4, two arms per session)

- [x] Base model: **Qwen2.5-0.5B-Instruct** (~2.4 h per 10K arm vs ~5–6 h for Llama-3.2-1B; matches the pilot). IFD redone with it in Stage 2b (2026-10-30)
- [ ] Switch to a second Kaggle account: new API token, upload the v4 arms as a private dataset `bn-full-arms` (steps in HANDOFF.md)
- [ ] Write the Stage 5 kernel: adapt `scripts/kaggle_lora_training.py` to run two arms at once (one per GPU), with checkpoint resume
- [ ] Train B, C (seed 42), ~2.4 h each, both in one session
- [ ] Train D, E, F (seed 42)
- [ ] Extra seeds for B and C (2 more each), for confidence intervals on the main comparison
- [ ] Arm A (whole pool): **decide** multi-session full run (~22 h) or a compute-matched run (same steps as a 10K arm)

## 4. Evaluation (Stage 6)

- [ ] Held-out loss per model on the 470 held-out rows (adapt `scripts/heldout_per_example.py`)
- [ ] Paired bootstrap vs B and vs base (`scripts/compare_arms_bootstrap.py`), loss by domain
- [ ] **Decide on downstream benchmarks.** Held-out loss alone is weak evidence for a paper. Candidates: Bengali MMLU (already in decontamination), mHumanEval-bn, a small generation eval judged pairwise
- [ ] Run the downstream benchmarks for every model
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
| Kaggle account | stages 1–4: habiburrahman153; Stage 5+: second account (TBD) |
| Arm A: full or compute-matched | open |
| Downstream benchmarks | open |
