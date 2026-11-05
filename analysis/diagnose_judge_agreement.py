"""Diagnose the weak judge-vs-human quality agreement (ROADMAP phase 0).

Questions:
  1. Is it the Qwen judge specifically, or would any LLM judge do this? The Stage 9
     independent rater (Llama-3.1-8B) scored the same rows, so its agreement with the
     human labels can be compared directly.
  2. Is the judge's quality score simply saturated (no discriminative power)? Compare the
     spread of the judge scores with the spread of the human labels.
  3. Does the fresh Stage 9 pass agree with the production Stage 3 pass? The 29 rows that
     reuse the production score give a direct check on the two runs.
  4. Is the human label maybe judging something different (e.g. factual correctness that
     the judge cannot see)? Look at the rows where human and judge disagree most.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "results/full_run/judge_human_validation.csv"

df = pd.read_csv(VAL)
print(f"rows: {len(df)}\n")

print("=" * 70)
print("1. WHO AGREES WITH THE HUMAN?")
print("=" * 70)
for metric, human_col in (("complexity", "human_complexity"), ("quality", "human_quality")):
    h = df[human_col].astype(float)
    for name, col in (("Qwen judge (Stage 3 rubric)", f"llm_{metric}"),
                      ("Llama rater (1-5 usefulness)", "llama_usefulness")):
        if col not in df.columns:
            continue
        x = df[col].astype(float)
        if metric == "quality" and name.startswith("Llama"):
            pass  # Llama rates usefulness on a 1-5 scale already
        mask = h.notna() & x.notna()
        rho = spearmanr(h[mask], x[mask]).statistic
        print(f"{metric:11s} vs {name:32s} rho={rho:+.4f}  (n={int(mask.sum())})")
    print()

print("=" * 70)
print("2. HOW MUCH DISCRIMINATION IS THERE? (spread of the scores)")
print("=" * 70)
for metric, human_col in (("complexity", "human_complexity"), ("quality", "human_quality")):
    h = df[human_col].astype(float)
    for name, col in (("human", human_col), (f"Qwen {metric}", f"llm_{metric}"),
                      ("Llama usefulness", "llama_usefulness")):
        x = df[col].astype(float)
        print(f"{metric:11s} {name:20s} mean={x.mean():.3f} sd={x.std():.3f} "
              f"min={x.min():.2f} max={x.max():.2f} "
              f"unique={x.nunique()}")
    print()

print("=" * 70)
print("3. DOES THE FRESH PASS MATCH THE PRODUCTION PASS?")
print("=" * 70)
pool = df[df["source"] == "stage3_pool"]
fresh = df[df["source"] == "stage9_fresh"]
print(f"production-sourced rows: {len(pool)} | fresh-sourced rows: {len(fresh)}")
print(f"production rows judge quality: mean={pool['llm_quality'].mean():.3f} sd={pool['llm_quality'].std():.3f}")
print(f"fresh rows judge quality:      mean={fresh['llm_quality'].mean():.3f} sd={fresh['llm_quality'].std():.3f}")
for metric, human_col in (("complexity", "human_complexity"), ("quality", "human_quality")):
    sub = pool
    h = sub[human_col].astype(float)
    x = sub[f"llm_{metric}"].astype(float)
    rho = spearmanr(h, x).statistic
    print(f"production rows only: {metric} rho={rho:+.4f} (n={len(sub)})")
print()

print("=" * 70)
print("4. THE 12 ROWS WHERE HUMAN AND JUDGE DISAGREE MOST ON QUALITY")
print("=" * 70)
df["q_gap"] = df["human_quality"].astype(float) - df["llm_quality"].astype(float)
worst = df.reindex(df["q_gap"].abs().sort_values(ascending=False).index).head(12)
for _, r in worst.iterrows():
    print(f"\nid={int(r['id'])}  human_q={int(r['human_quality'])}  judge_q={r['llm_quality']:.2f}  "
          f"llama={r['llama_usefulness']:.2f}")
    print(f"  I: {str(r.get('instruction', ''))[:110]}")
    if "reason" in df.columns:
        print(f"  human reason: {str(r['reason'])[:180]}")
    else:
        print("  (no human reason in this file)")

print("\n" + "=" * 70)
print("5. DOES THE JUDGE SEPARATE HUMAN-GOOD FROM HUMAN-BAD AT ALL?")
print("=" * 70)
for metric, human_col in (("complexity", "human_complexity"), ("quality", "human_quality")):
    h = df[human_col].astype(float)
    x = df[f"llm_{metric}"].astype(float)
    for threshold in (2, 3, 4):
        low = x[h <= threshold]
        high = x[h > threshold]
        if len(low) and len(high):
            print(f"{metric:11s} human<={threshold}: judge mean={low.mean():.3f} (n={len(low)}) | "
                  f"human>{threshold}: judge mean={high.mean():.3f} (n={len(high)}) | "
                  f"gap={high.mean() - low.mean():+.3f}")
    print()
