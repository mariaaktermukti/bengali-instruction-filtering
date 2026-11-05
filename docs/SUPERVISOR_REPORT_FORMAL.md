# Progress Report: LLM-Judged Subset Selection for Bengali Instruction Tuning

**Project:** Selecting a 10K subset from a 100K Bengali instruction pool using LLM-judged complexity × quality with a diversity step (DEITA-style)

**Base model:** Qwen2.5-0.5B-Instruct (LoRA)

**Report date:** 2026-11-05
**Status:** All experimental stages complete (Stages 1–9). Paper writing is the remaining phase.

---

## 1. Research Question

For Bengali instruction tuning, does selecting a 10K subset by LLM-judged complexity × quality (with a diversity step, DEITA-style) outperform random selection, top-quality-only selection, or IFD (Cherry-LLM)? How close does such a subset get to training on the entire clean pool?

**Hypotheses tested.** H1: complexity × quality selection beats random 10K. H2: the diversity step adds measurable value over score-only selection. H3: IFD (Cherry-LLM) is competitive on compute-matched grounds. H4: the LLM judge correlates with human judgement.

---

## 2. Method

### 2.1 Data pipeline

Source dataset: `md-nishat-008/Bangla-Instruct` (342,390 rows, MIT licence).

| Step | Detail | Result |
|---|---|---|
| Sampling | Domain-stratified, seed 42 | 100,000 pool rows + 500 held-out |
| Held-out split | Removed **before** any filtering; never trained on | 500 → **470** after the same filters |
| Normalisation | ftfy + NFC; preserves ZWJ/ZWNJ, fenced code byte-identical, cleans HTML entities and stray whitespace | — |
| Heuristic filters | Length, mojibake, template artefact, language ratio, translationese, echo, truncation, character spam, repetition, identity refusal | **5,081 dropped (5.1%)** — language 4,215, translationese 478, repetition 128, template 87, truncation 67, length 56, identity refusal 28, others 22 |
| Deduplication | Exact 2,991 + MinHash-LSH (128 perms, Jaccard ≥ 0.80, 5-gram shingles) 736 near-dup pairs | — |
| Decontamination | 13-gram overlap (8-gram for short texts) against Bengali MMLU, BanglaRQA, mHumanEval-BN, plus the held-out prompts; n-grams shared by > 10 pool rows treated as templates, not leaks | **71 dropped** |
| Final | | **Clean pool: 91,120 rows** (111.5M tokens) |

**Filter-bias check** (`results/full_run/domain_filter_bias.csv`). Two domains are hit much harder than the rest:

| Domain | Drop rate | Main reason | Pool share | Arm C | Arm E | Arm F |
|---|---|---|---|---|---|---|
| mcq_classification | 21.0% | translationese (366) | — | — | — | — |
| code | 20.7% | language (3,038) | 11.4% | 3.0% | 1.4% | 21.2% |
| open_qa | 2.1% | language (620) | 37.7% | 47.3% | 52.2% | 42.5% |
| math | 1.3% | language (344) | 36.8% | 37.8% | 36.9% | 20.7% |

Likely cause: code outside ``` fences counts as English prose. The held-out set is filtered identically, so the arm comparison stays internally fair, but the code under-representation in arms C and E is a real confound for code prompts.

### 2.2 Scoring

- **LLM judge:** Qwen2.5-7B-Instruct, fp16, vLLM tensor-parallel over 2×T4, digit-expectation scoring (probability mass over digits 1–6 → continuous score), English rubric, parse rate 1.00 over all 91,120 rows. 3 sessions, ~21 GPU-hours. The rubric was extracted verbatim from the pilot notebook and reused byte-identically in Stage 9.
- **Backend validation (Stage 3a):** vLLM TP=2 vs the HF pilot implementation — **Spearman 1.000**, 4,374 rows/h. (AWQ was 10% faster but changed the ranking, MAE 0.20, so it was rejected.)
- **IFD:** recomputed with Qwen2.5-0.5B-Instruct (the model actually being fine-tuned), 91,120 rows, 0 NaN, Spearman 0.68 vs the original Llama-based IFD.
- **Embeddings:** BGE-M3, for the diversity step.

### 2.3 Arms

| Arm | Selection rule | Rows | Tokens (Llama-tok) | Tokens (Qwen-tok, actually trained) |
|---|---|---|---|---|
| A | Whole clean pool | 91,120 | 111.5M | 102.4M |
| B | Random | 10,000 | 12.2M | 11.2M |
| C | DEITA: c × q, greedy with diversity filter | 10,000 | 15.5M | 14.2M |
| D | Quality only (judge q, top-K) | 10,000 | 15.5M | 14.2M |
| E | c × q, **no** diversity step | 10,000 | 15.2M | 13.9M |
| F | Cherry-LLM / IFD | 10,000 | 10.9M | 10.1M |

> **Token-accounting note.** Stage 1 counted tokens with the Llama-3.2-1B tokenizer (the then-default base model), and `results/full_run/token_efficiency.csv` is built on those counts. The training stage tokenised with the real Qwen2.5-0.5B tokenizer, which yields ~9% fewer tokens. Both columns are given above; the ordering and every comparison in this report are unaffected, but the paper should state which tokenizer each table uses.

**τ calibration.** The pilot's τ = 0.90 left C 91.5% identical to E, i.e. the diversity step was inert. A sweep over τ ∈ [0.65, 0.96] selected **τ = 0.74** (≈ pool NN-similarity p25 = 0.746), giving C∩E = 58.1%, the lowest mean pairwise similarity (0.379 vs E's 0.391), and a code share of 3.0% vs E's 1.4%. Arms were rebuilt at τ = 0.74 (Stage 4 v3); arm F was rebuilt with the Qwen IFD (Stage 4 v4).

### 2.4 Training and evaluation

- **Training:** LoRA on Qwen2.5-0.5B-Instruct, 16 runs (arms B–F × 3 seeds, arm A × 1). Seed spread ≈ 0.001 on held-out loss.
- **Held-out loss:** 470 rows, per-example, written by the training stage.
- **Paired bootstrap:** per model, and by domain, token-weighted.
- **Downstream:** Bengali MMLU (translated) and BanglaRQA (native), both decontaminated against the pool. Answers are prefilled; greedy decoding with repetition penalty 1.0 (Qwen's default 1.1 penalises Bengali byte-level pieces).
- **Pairwise judging:** 470 held-out prompts × 21 model pairs × 2 orders = **19,740 comparisons**, 100% complete. Judge Qwen2.5-7B-Instruct, same backend as Stage 3, with position-bias correction by averaging both presentation orders.
- **Qualitative samples:** `results/full_run/qualitative_samples.md`.

---

## 3. Results

### 3.1 Main table

Loss lower is better; win rate higher is better. Loss columns show the delta vs arm B.

| Model | Tokens (M) | Held-out loss | Δ loss vs B | Token-wt loss | MMLU acc | RQA F1 | Pairwise win vs B (95% CI) |
|---|---|---|---|---|---|---|---|
| base | — | 0.9779 | +0.1652 | 0.9386 | 0.2583 | 0.2071 | 0.4729 [0.449, 0.498] |
| A | 111.5 | 0.7169 | −0.0957 | 0.7188 | 0.2624 | 0.1827 | 0.7032 [0.675, 0.731] |
| B | 12.2 | 0.8126 | — | 0.8172 | 0.2584 | 0.1504 | 0.5000 [0.500, 0.500] |
| C | 15.5 | 0.8380 | +0.0254 | 0.8186 | 0.2505 | 0.1757 | 0.3761 [0.352, 0.406] |
| D | 15.5 | 0.8572 | +0.0446 | 0.8307 | 0.2666 | 0.1597 | 0.4638 [0.436, 0.491] |
| E | 15.2 | 0.8661 | +0.0535 | 0.8255 | 0.2370 | 0.1681 | 0.3484 [0.321, 0.377] |
| F | 10.9 | 0.8260 | +0.0133 | 0.8327 | 0.2358 | 0.1531 | 0.3207 [0.294, 0.349] |

**Benchmark deltas vs B** (from `results/full_run/main_results_table.csv`):

| Model | Δ MMLU | Δ RQA F1 |
|---|---|---|
| base | −0.0001 | +0.0567 |
| A | +0.0040 | +0.0323 |
| C | −0.0079 | +0.0253 (best 10K arm, p = 1.0) |
| D | **+0.0082** (p = 0.999 vs B; not significant vs base or A) | +0.0093 |
| E | −0.0213 | +0.0177 |
| F | −0.0226 | +0.0027 (CI crosses 0) |

**Head-to-head pairwise** (all 21 pairs, 19,740 comparisons, 0 missing): C beats E (0.6170) and F (0.6612); D beats E (0.7585) and F (0.7633); C vs D is statistically tied (0.4947 vs 0.5053, p = 0.339); A dominates every arm (win rate 0.63–0.79; the narrowest is A vs D at 0.6346).

### 3.2 What the results show

**(a) The full pool remains the strongest arm.** Arm A wins on held-out loss, token-weighted loss, and pairwise preference. The honest framing is that a 10K subset does not approach full-pool performance — and none of the intelligent selection methods beat random on held-out loss or pairwise preference.

**(b) Complexity × quality selection did not beat random (H1 not supported).** Arm C has a *higher* held-out loss than arm B (0.8380 vs 0.8126) and loses the head-to-head pairwise comparison (0.3761). Arm E is worse still (0.8661). Arm F (IFD) also loses (0.3207), though it reaches a better held-out loss than C/D/E while using the fewest tokens.

**(c) The diversity step is the one component with a clear, measurable effect (H2 supported).** C vs E, differing *only* in the τ filter: held-out loss 0.8380 vs 0.8661 (−0.0281), token-weighted loss −0.0069, MMLU +0.0135, RQA +0.0076, and head-to-head pairwise 0.6170 for C. The diversity filter also raises code representation (3.0% vs 1.4%), lowers mean pairwise embedding similarity (0.379 vs 0.391), raises distinct-2 response tokens (0.503 vs 0.451), and improves embedding-space coverage (0.746 vs 0.727).

**(d) Token efficiency.** Loss reduction per million training tokens (Llama-tokenizer counts): A 0.0023, B 0.0136, **F 0.0139**, **C 0.0090**, D 0.0078, E 0.0074. Read carefully: F and B score highest per token because they train on the fewest tokens, not because they reach a better absolute loss — B still has the best absolute held-out loss of the 10K arms. Within the score-based family (C, D, E), C is clearly the most token-efficient, which isolates the **diversity step** as the active ingredient rather than the c × q score.

**(e) Most of the gap is one domain: code.** Decomposing the held-out loss gap per domain (`results/full_run/heldout_by_domain.csv`) shows that the C − B regression of +0.0254 is **78.4% code**. Code is 70 of the 470 held-out rows (14.9%), and C's loss on them is 0.9950 against B's 0.8614 (+0.1336). On the other 400 rows the gap is only **+0.0055**. The same holds for E (79.1% code) and inverts for F (F's code share is 21.2%, and F's code loss 0.8423 is *below* B's 0.8614 — F is the only arm that improves on code, and it wins there: 0.4643 vs B on 70 code prompts). This is the mechanical consequence of §2.1: c × q systematically under-selects code, and the loss penalty lands almost entirely there.

**(f) The downstream benchmarks are more nuanced than the loss numbers.**
- **Arm D has the highest MMLU accuracy of every model trained here** (0.2666, +0.0082 vs B, p = 0.999), but this is a *nominally* best result, not a demonstrated one: a paired bootstrap over all 14,042 questions gives D − base = +0.0083, 95% CI [−0.0009, +0.0174], p = 0.964, and **D − A = +0.0042, 95% CI [−0.0038, +0.0121], p = 0.849**. D is not significantly better than the base model or than arm A. It is also the arm selected purely on the judge's quality score, which §3.4 shows is exactly the dimension the judge cannot verify against humans.
- **Arm C has the best RQA F1 of the 10K arms** (0.1757, +0.0253 vs B, p = 1.0), and this one *does* survive seed variation — every C seed (0.1684–0.1800) is above every B seed (0.1399–0.1560).
- On BanglaRQA, **the un-finetuned base model scores highest of all** (0.2071; on answerable questions only, 0.2744). Every fine-tuned arm is below it. On a native-Bengali reading-comprehension benchmark, fine-tuning on this synthetic pool costs accuracy.

### 3.3 Seed robustness, and what the generations actually look like

**The seed spread on the downstream benchmarks is larger than several of the arm gaps.** Held-out loss is stable (spread ≈ 0.001), but MMLU and RQA are not:

| Arm | MMLU per seed | Range | RQA F1 per seed | Range |
|---|---|---|---|---|
| A | 0.2624 | — | 0.1826 | — |
| B | 0.2522 / 0.2639 / 0.2591 | 0.0117 | 0.1399 / 0.1560 / 0.1554 | 0.0161 |
| C | 0.2464 / 0.2465 / 0.2585 | 0.0121 | 0.1684 / 0.1800 / 0.1787 | 0.0116 |
| D | 0.2601 / 0.2690 / 0.2706 | 0.0105 | 0.1595 / 0.1454 / 0.1741 | 0.0287 |
| E | 0.2337 / 0.2383 / 0.2391 | 0.0054 | 0.1585 / 0.1786 / 0.1675 | 0.0201 |
| F | 0.2376 / 0.2339 / 0.2359 | 0.0037 | 0.1379 / 0.1642 / 0.1571 | 0.0263 |

Consequences for what the paper may claim:
- **C's RQA advantage over B is robust** — every C seed beats every B seed.
- **D's MMLU advantage is not robust** — D's seed range (0.2601–0.2706) overlaps B's (0.2522–0.2639), and the paired bootstrap against base and against A both cross zero.
- **E's and F's RQA results are entangled** — their seed ranges overlap C's and each other's, so the E < C ordering on RQA is not safe on RQA alone (it is safe on the held-out loss and the pairwise judging).
- Every downstream comparison should therefore be reported with the seed range next to it, not as a single mean.

**Generation quality: a degeneracy problem that the loss numbers hide.** The `hit_limit` column in `results/full_run/benchmarks/generation_stats.csv` is the fraction of the 470 held-out generations that ran into the 1,536-token cap without stopping:

| Run | Mean tokens | Hit the token limit | Bengali char share | Empty |
|---|---|---|---|---|
| A | 1,107.8 | 0.587 | 0.626 | 0 |
| B | 1,172.4 | 0.689 | 0.662 | 0 |
| D | 1,335.6 | 0.779 | 0.674 | 0 |
| base | 1,342.6 | 0.800 | 0.709 | 0 |
| F | 1,287.4 | 0.804 | 0.658 | 0 |
| E | 1,387.8 | 0.851 | 0.687 | 0 |
| **C** | 1,384.1 | **0.857** | 0.642 | 0 |

Arm A (the only arm that trains on 9× more data) produces the shortest and least truncated generations. **C is the worst offender** — 85.7% of its generations run to the token cap. The qualitative samples show exactly what that means: on the code prompt, C emits a Bengali "সংক্ষেপ:" section that repeats the instruction phrase dozens of times; on the list_brainstorm prompt it degenerates into "কেন কেন? কেন কেন?" repeated to the cap. Arms B, D and E show the same pattern; arm A and F produce clean, terminated code.

This is the mechanism behind the pairwise-judging result: the judge is asked to compare two often-truncated, repetitive generations, and it rewards the one that looks more complete. The paper must report the truncation rates alongside the pairwise win rates, or the pairwise comparison will overstate every arm's quality.

### 3.4 Judge validation against human labels (H4 — partially supported)

This is the most important addition of the current phase, and it explains (b) and (c).

100 rows were manually annotated for **complexity** and **quality** on a 1–5 scale (7 rows also carry a free-text reason). The judge emits 1–6 (expectation over digit probabilities), so judge scores are mapped onto the human scale with `1 + (judge − 1) × 4/5` and binned to the nearest integer. Spearman is invariant to this affine map.

29 of the 100 rows are in the production pool and reuse their Stage 3 score; the other 71 were scored fresh in Stage 9 with the identical rubric and backend. All 71 fresh rows cleared the probability-mass threshold.

| Comparison | n | Spearman ρ | p | Exact | Within ±1 | κ (quadratic) | MAE (human scale) |
|---|---|---|---|---|---|---|---|
| Complexity: judge vs human | 100 | **0.543** | <0.0001 | 40% | 95% | 0.316 | 0.717 |
| Quality: judge vs human | 100 | **0.078** | 0.44 | 14% | 54% | 0.033 | 1.562 |

- **Complexity agreement is usable:** ρ = 0.543 with 95% of ratings within one point.
- **Quality agreement is absent:** ρ = 0.078, p = 0.44, κ ≈ 0.03. The judge's quality score is also **ceilinged** — binned, 70 rows land on 4, 23 on 5, 7 on 3, and **none below 3**, while the human labels span 1–5.
- Restricting to the 29 production-scored rows alone gives complexity ρ = 0.591 and quality ρ = 0.174, so the null is not an artefact of the fresh pass.

**The quality failure is not model-specific.** An independent rater from a different family (`unsloth/Llama-3.1-8B-Instruct`, 1–5 overall usefulness) was run on the same 100 rows:

| Rater | ρ vs human complexity | ρ vs human quality |
|---|---|---|
| Qwen2.5-7B (Stage 3 rubric) | 0.543 | 0.078 |
| Llama-3.1-8B (independent) | 0.525 | −0.084 |

Both raters succeed on complexity and fail on quality. Note what this means: the rubric **does** put "correctness and factual accuracy" first in its quality criteria — this is not a missing instruction. The model simply does not enforce it. The score it actually produces behaves like a fluency / formatting / instruction-following judgment.

**Concrete failure cases.** On rows a human marked quality = 1 for a factual error, both raters score high:

| id | Human | Qwen judge | Llama | What the human objected to |
|---|---|---|---|---|
| 55 | q = 1 | 5.33 | 4.24 | "Great Albatross" listed as a Palearctic species (it is a Southern Ocean bird) |
| 83 | q = 1 | 5.27 | 4.35 | Generic film-criticism boilerplate that never answers the question |
| 20 | q = 1 | 5.12 | 4.36 | মঞ্চু সমাজ located in Bangladesh (it is in Jharkhand, India) |

And where a human gave a free-text reason for a low score, the pattern is consistent: id 4 (human q = 1, judge 3.00) — the response invents the actor "Dr. David Thompson" for a role played by John Simm and renames Colin Pitchfork as "Colin Richards"; id 7 (human q = 2, judge 5.00) — the output sentence is grammatically meaningless. The judge scored the second one a 5.

**Implication for the paper.** The judge's `c` is validated as a ranking signal; the judge's `q` is **not** validated against human quality. This must be stated as a limitation, and it directly weakens arm D (quality-only selection), which was selected on precisely the dimension the judge cannot verify — even though D has the best MMLU score in the table. It also qualifies arms C and E, which combine `c` with `q`.

### 3.5 Blind spot-check (independent, non-circular)

Stage 4 produced a sheet of 100 items — 50 from arm C, 50 from arm B — with the arm label hidden. Rating it with the Qwen judge would be **circular**, because arm C was selected to maximise that judge's c × q; such a rating would reward the selection criterion rather than test it. The sheet was therefore rated by the independent Llama-3.1-8B rater on a 1–5 overall-usefulness scale (the same scale as the sheet's human `rating_1to5` column). The two arms hold 50 *different* items each, so this is an unpaired two-sample comparison.

| Arm | n | Independent mean | 95% CI | Qwen c × q (circular) |
|---|---|---|---|---|
| B | 50 | 4.055 | [4.005, 4.101] | 17.60 |
| C | 50 | 4.124 | [4.088, 4.164] | 23.25 |

**Mean C − mean B = +0.070** (95% CI [+0.009, +0.133]), p(C better) = 0.990.

Under an independent rater, arm C's rows are better than arm B's by a small but statistically significant margin. The circular Qwen view overstates the gap (23.25 vs 17.60) by roughly an order of magnitude. Both numbers are reported; the independent one is the headline.

The human half of the spot-check remains open and optional: a human rater can fill `results/full_run/spotcheck_ratings.csv` and `analysis/join_spotcheck.py` will fold it into the same table.

---

## 4. Contribution and honest limitations

**Contribution.** A negative-result contribution with a validated explanation: on Bengali instruction data, LLM-judged complexity × quality selection does **not** beat random 10K selection, and a human-validated judge diagnostic identifies *why* — the quality rubric asks for correctness but the model does not enforce it, so the selection signal is partly orthogonal to true data quality. The DEITA-style diversity step is the one component that demonstrably helps. This directly contradicts several published "quality selection works" claims.

**Limitations to report.**

1. **Judge quality score is unvalidated** (ρ = 0.078 vs humans) and ceilinged; arm D's premise is undermined even though arm D has the nominally highest MMLU accuracy.
2. **Synthetic source data** — Bangla-Instruct was generated by GPT-4o and Claude-3.5.
3. **Single base model** — Qwen2.5-0.5B-Instruct only; arm A has a single seed.
4. **Filter bias on code** — the language filter removes 20.7% of code rows, and code representation varies 1.4%–21.2% across arms. This is not a side note: **code alone accounts for 78.4% of the C − B held-out loss gap**, so the "c × q selection does not beat random" result is largely a code-representation effect.
5. **Judge-family dependence in the selection criterion** — arm C is selected by a Qwen-family judge; the independent spot-check mitigates but does not eliminate this.
6. **Human validation sample is 100 rows** — enough to detect a null (quality) and a moderate effect (complexity), but not for fine-grained calibration.
7. **Token accounting uses two tokenizers** — Stage 1 counts (Llama-3.2-1B) and training counts (Qwen2.5-0.5B) differ by ~9%; each table must state which it uses.
8. **BanglaRQA F1 is below the un-finetuned base model for every arm** — the downstream claim for this data is weaker than the loss numbers suggest.
9. **Seed variance on the downstream benchmarks exceeds several arm gaps** — MMLU seed ranges are 0.004–0.012 and RQA ranges 0.012–0.029, against arm gaps of the same order. Only the held-out loss, the token-weighted loss and C-vs-B on RQA are robust to seed choice; arm D's MMLU lead is not.
10. **Generation degeneracy is high and unmeasured by the loss metrics** — 59%–86% of generations hit the 1,536-token cap without stopping, and arm C is the worst at 85.7%. The pairwise judge compares these often-truncated outputs, so pairwise win rates are likely biased toward whichever arm produces longer, more complete-looking text. The pairwise numbers should be reported with the truncation rates, or a truncation-aware re-scoring should be run.
11. **BanglaRQA contains 5 duplicated question ids** (1,493 rows, 1,488 unique ids); the paired bootstrap drops them to keep the comparison aligned across runs, so the RQA tables are computed on 1,488 questions. This is a property of the upstream dataset, not of the arms, but it should be stated.

---

## 5. Work completed and remaining

**Completed.** Stages 1–9: data pipeline, scoring, arm selection, 16 LoRA training runs, held-out evaluation with paired bootstrap, Bengali MMLU and BanglaRQA benchmarks, 19,740 pairwise comparisons, qualitative samples, Phase 5 analysis tables, and (this phase) judge-vs-human validation plus the independent blind spot-check. Total compute ≈ 60–90 GPU-hours across Kaggle accounts; Stage 9 itself took 20.9 minutes.

**Remaining — Phase 6, writing.** Outline and target venue; introduction and related work (DEITA, Cherry-LLM/IFD, TigerLLM, Bangla-Instruct); data and pipeline; selection methods; experimental setup; results and analysis; limitations; abstract, conclusion and final pass.

**Optional extensions.**
- Human spot-check pass (1–2 hours) to strengthen the arm C claim.
- A revised quality rubric, and a second judge that is asked to verify claims rather than rate fluency, to test whether the judge can be repaired and the arms rebuilt (2–3 GPU-days).

---

## 6. Artefacts

| Item | Path |
|---|---|
| Master results table | `results/full_run/main_results_table.{csv,md}` |
| Token efficiency | `results/full_run/token_efficiency.csv` |
| Diversity ablation (C vs E) | `results/full_run/diversity_ablation.csv` |
| Domain mix & filter bias | `results/full_run/domain_filter_bias.csv` |
| Held-out loss by domain | `results/full_run/heldout_by_domain.csv` |
| Per-seed downstream runs | `results/full_run/benchmarks/mmlu_per_run.csv`, `rqa_per_run.csv` |
| Generation length & truncation | `results/full_run/benchmarks/generation_stats.csv` |
| τ sweep | `results/full_run/tau_sweep_summary.csv` |
| Judge vs human validation | `results/full_run/judge_human_validation.md` |
| Blind spot-check | `results/full_run/spotcheck_results.md` |
| Judge validation summary | `results/full_run/judge_validation_summary.csv` |
| Qualitative samples | `results/full_run/qualitative_samples.md` |
| Reproducible analysis | `analysis/generate_phase5_analysis.py`, `analysis/validate_llm_judge.py`, `analysis/join_spotcheck.py`, `analysis/diagnose_judge_agreement.py` |
| Stage 9 kernel | `kaggle/full_run/s09_judge_human/` |
| Run log and decisions | `HANDOFF.md`, `ROADMAP.md`, `kaggle/full_run/PLAN.md` |
