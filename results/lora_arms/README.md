# LoRA experiments on the selection arms: results and conclusions

Draft of the final thesis chapters, written from the results in this folder (2026-10-26).

## Experimental setup

We trained one small model per data subset ("arm"), kept every other setting identical, and compared held-out loss. The arms come from the Iteration 03 pilot of our selection pipeline.

### Data

The pilot is a stratified sample of 1,000 rows from Bangla-Instruct (`md-nishat-008/Bangla-Instruct`). We normalised the Bengali text, applied heuristic filters, removed exact and MinHash near-duplicates, and dropped anything sharing a 13-gram with a benchmark. That left a clean pool of 955 rows. Another 47 rows, sampled before filtering and never used for training, are the held-out set.

An LLM judge (Qwen2.5-7B-Instruct) scored every pool row for instruction complexity *c* and response quality *q*, on a 1–6 scale. Each score is the expected value over the judge's probabilities for the digits, not a single sampled digit. We also computed instruction-following difficulty (IFD) with Llama-3.2-1B and embeddings with bge-m3.

| Arm | Rows | Selection rule |
| --- | --- | --- |
| A | 955 | Whole clean pool, no selection (anchor) |
| B | 400 | Uniform random |
| C | 400 | Rank by *c* × *q*, greedy add if cosine similarity to every selected row < τ = 0.9 (DEITA-style) |
| D | 400 | Top 400 by *q* only |
| E | 400 | Top 400 by *c* × *q*, no diversity step |
| F | 400 | Top 400 by IFD among rows with IFD < 1 (Cherry-LLM baseline) |

We did not train arm E. It turned out to hold the same 400 rows as arm C, in the same order; the Discussion explains why.

### Model and training

| Setting | Value |
| --- | --- |
| Base model | Qwen2.5-0.5B-Instruct |
| Adapter | LoRA, r = 8, α = 16, dropout 0.05, on q, k, v, o projections (1.08M trainable parameters, 0.22%) |
| Prompt format | Qwen chat template; loss on response tokens only |
| Max sequence length | 3,072 tokens (covers 99% of arm C rows) |
| Optimiser | AdamW, learning rate 1e-4, linear decay, 1 epoch |
| Batch | 1 per device × 4 gradient accumulation = 4 |
| Precision | fp32 weights, fp16 mixed precision, gradient checkpointing |
| Seed | 42 |
| Hardware | Kaggle, one NVIDIA Tesla T4 (16 GB) |

### Evaluation

Our main metric is the mean cross-entropy over response tokens on the 47 held-out rows, reported alongside perplexity. We measure it once before training and once after. Before training it is simply the base model's loss, because a fresh LoRA adapter changes nothing.

We also looked at greedy generations for four prompts: two Bengali prompts we wrote, plus the first two held-out instructions.

## Results

Training on any arm lowered held-out loss. Choosing *which* 400 rows to train on did not help: no selection rule beat a random 400. The one arm that beat random was A, which used all 955 rows and came out 0.020 nats per response token lower.

| Arm | Rows | Steps | Train loss | Held-out loss | Held-out perplexity | Change vs base | Training time (min) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base (no training) | – | – | – | 0.958 | 2.606 | – | – |
| A | 955 | 239 | 0.902 | 0.881 | 2.414 | −0.077 | 13.5 |
| B | 400 | 100 | 0.925 | 0.901 | 2.463 | −0.057 | 5.4 |
| C | 400 | 100 | 0.917 | 0.902 | 2.465 | −0.056 | 5.8 |
| D | 400 | 100 | 0.869 | 0.907 | 2.476 | −0.051 | 5.9 |
| F | 400 | 100 | 1.002 | 0.910 | 2.483 | −0.048 | 6.4 |

Each arm's training loss is measured on its own data, so the column cannot be compared across rows. Arm D is still worth pointing out. It has the lowest training loss of any arm and a worse held-out loss than random. Our reading is that the rows the judge liked best are easier to fit, not more useful to learn from.

The gain over the base model is real for every arm: all five 95% intervals sit between −0.100 and −0.037. GPU noise is small, too. We ran arm C twice and the held-out loss moved by 0.0004, well below the gaps between arms.

The table below compares each arm with random selection, paired over the same 47 held-out rows. Intervals come from 10,000 bootstrap resamples (seed 42). Negative means better than random.

| Arm | Difference vs random | 95% interval | Share of resamples below random |
| --- | --- | --- | --- |
| A: full pool, 955 rows | −0.0200 | [−0.0249, −0.0157] | 100% |
| C: *c* × *q* + diversity, 400 | +0.0006 | [−0.0016, +0.0029] | 31.5% |
| D: top quality *q*, 400 | +0.0052 | [−0.0001, +0.0129] | 2.8% |
| F: top IFD, 400 | +0.0082 | [+0.0021, +0.0158] | 0.2% |

We cannot tell arm C apart from random. Arm D looks slightly worse, though its interval just reaches zero. Arm F is worse: only 0.2% of resamples put it below random.

Broken down by task, the code rows separate the arms most, though there are only 7 of them. Loss on code was 0.740 for A, 0.773 for B and C, 0.786 for D and 0.808 for F. On the 18 math rows, C edged out the other 400-row arms (0.763 against 0.767 for B).

The sample generations point the same way. After training, every arm wrote a correct Python `is_prime` function; the base model had answered the same prompt with unrelated Java. But every arm still made up a name for Bangladesh's national flower. Every arm also answered the held-out geometry problem in English and treated the ellipse as a hyperbola, and every arm fell into repetition on the restaurant-description prompt.

## Discussion

In this pilot, more data beat better-chosen data. Going from 400 to 955 rows lowered held-out loss by 0.020. The best selection rule moved it by less than 0.001.

### Why the selection rules did not separate

Looking back, the pilot gave the selection rules little room to differ, for three reasons.

First, the diversity step never removed anything. At τ = 0.9, no two of the top-ranked rows were that similar; the average pairwise cosine similarity inside arm C is 0.39. So arm C is exactly arm E, the top 400 rows by *c* × *q*. What we tested was score-based ranking, not the DEITA diversity filter.

Second, the arms overlap a lot. Every arm takes 400 of the 955 rows, 42% of the pool, so any two arms share between 165 and 208 rows. Two random 400-row draws would share about 168. Roughly half of each arm is the same data as any other arm.

Third, the pool had already been cleaned. Heuristic filtering, deduplication and decontamination all ran before selection, and the judge rated what was left highly (mean *q* = 4.70 out of 6). There was not much bad data left for a quality score to throw out.

### What the score-based rules selected

Ranking by quality (D) or by IFD (F) changed the mix of task types, and both arms did worse than random. Arm D pushed math from the pool's 35% to 44% and cut open QA from 38% to 28%. Arm F went the other way: 50% open QA and 24% math. Both lost the most ground on held-out code. Because the held-out set follows the pool's original mix, any rule that tilts the mix away from it pays for it in this metric.

Arm C also tilted toward math (43%) but kept open QA at the pool's 38%, with 10% code, and it ended up level with random. On held-out math it was narrowly the best of the 400-row arms. That fits the reasoning behind DEITA, that scoring complexity alongside quality favours harder examples. Still, a 0.004 difference on 18 rows is not evidence of anything.

### What this means for the thesis question

The pilot does not support our hypothesis that LLM-judge selection, keeping 42% of the data, helps a small Bengali model more than random selection does. What it does show is that the pipeline works end to end. The judge returned a usable score for every row. Its English and Bengali rubrics agreed closely (Spearman 0.89 for complexity, 0.80 for quality). And every trained arm improved on the base model. The real test is the full 100,000-row pool, where selection keeps a much smaller share and the arms can actually differ.

## Limitations

This was a feasibility pilot. The results say more about whether the pipeline runs than about how the selection methods compare at scale.

- The pool is small. Each arm keeps 42% of 955 rows. At the planned 100,000 rows, a subset of 400 or even 10,000 rows is a much stronger filter.
- The held-out set is small. With 47 rows, the bootstrap intervals (roughly ±0.003 to ±0.008 nats) are wide compared with the gaps between arms. The per-domain numbers rest on 18 math, 14 open-QA and 7 code rows.
- Each arm was trained once, with seed 42. The bootstrap only accounts for which held-out rows were sampled, not for training randomness. Our repeated arm C run moved by 0.0004, which hints that seed variance is low, but we have not measured it properly.
- Loss is only a proxy. It measures how well the model predicts the reference answers, not whether its answers are correct, follow the instruction or read as fluent Bengali. The generations show those can stay poor while loss falls.
- The model is small and the training short. Qwen2.5-0.5B-Instruct knows little Bengali, and one epoch with a rank-8 adapter barely moves it. A larger model might react to data quality differently.
- The diversity filter was never really tested, since τ = 0.9 removed nothing.
- The evaluation is in-distribution. The held-out rows come from the same Bangla-Instruct data as the training pool, so the metric rewards matching that distribution rather than general ability.

## Conclusion and future work

We built a pipeline that filters Bengali instruction data, scores it with an LLM judge, selects subsets and fine-tunes a model on each, and ran it end to end on a 955-row pilot. Every subset improved held-out loss on Qwen2.5-0.5B-Instruct. None of the selection rules beat random selection at the same size, and IFD-based selection did significantly worse. Training on all 955 rows beat every 400-row subset.

Along the way we fixed the problems that had kept the study from running on free Kaggle GPUs. The preinstalled torchao package crashed LoRA, and a 512-token limit cut off 351 of the 400 Bengali training rows. With those fixed, a 400-row arm trains in about 6 minutes on a T4.

Next steps, most important first:

1. Run the full pool. Score and select from all 100,000 rows. Extrapolating from the pilot, the LLM judge alone needs about 94 GPU-hours, so this needs vLLM batching or several Kaggle sessions. Keep 10% of the data or less, where selection has a chance to matter.
2. Recalibrate τ. Derive the diversity threshold from the pool's nearest-neighbour similarities (its 90th percentile, say) so that arm C actually differs from arm E.
3. Evaluate on more, and broader, data. Hold out at least 500 rows, and add the Bengali benchmarks we already decontaminate against: Bengali MMLU, BanglaRQA and mHumanEval-Bengali.
4. Train each arm with three seeds and report the spread across seeds next to the bootstrap interval.
5. Try a bigger model. Repeat the best arm and the random arm on a 1–3B model to see whether selection matters more once the base model knows more Bengali.

## Reproducibility

Everything needed to reproduce these results is in this branch: the code, each arm's training report and the evaluation outputs. [KAGGLE_EXPERIMENTS.md](../../KAGGLE_EXPERIMENTS.md) has the rerun steps.

The Kaggle notebooks are private to the habiburrahman153 account.

| Notebook | Role |
| --- | --- |
| [bangla-filtering-selection](https://www.kaggle.com/code/habiburrahman153/bangla-filtering-selection) | Selection pipeline; produces arms A–F and the held-out set |
| [bengali-lora-arm-a](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-a) | LoRA training, arm A |
| [bengali-lora-arm-b](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-b) | LoRA training, arm B |
| [bengali-lora-arm-c](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-c) | LoRA training, arm C |
| [bengali-lora-arm-d](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-d) | LoRA training, arm D |
| [bengali-lora-arm-f](https://www.kaggle.com/code/habiburrahman153/bengali-lora-arm-f) | LoRA training, arm F |
| [bengali-lora-heldout-eval](https://www.kaggle.com/code/habiburrahman153/bengali-lora-heldout-eval) | Per-example held-out loss for the base model and all adapters |

The Kaggle runs used PyTorch 2.11.0 (CUDA 12.8) on Python 3.13, with Kaggle's preinstalled transformers and peft. The bootstrap used 10,000 resamples with seed 42 ([`scripts/compare_arms_bootstrap.py`](../../scripts/compare_arms_bootstrap.py)).

| File in this folder | Contents |
| --- | --- |
| [`summary.csv`](summary.csv) | One row per arm: rows, steps, train loss, held-out loss and perplexity, runtime |
| [`heldout_comparison.csv`](heldout_comparison.csv) | Paired bootstrap against the base model and arm B |
| [`heldout_by_domain.csv`](heldout_by_domain.csv) | Held-out loss by task domain |
| [`heldout_per_example.csv`](heldout_per_example.csv) | Per-example losses for the base model and every arm |
| `arm_*_training_report.json` | Config, token stats, loss curve, before/after generations |
