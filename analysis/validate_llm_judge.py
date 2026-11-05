"""Judge-vs-human validation on the 100 human-annotated rows (ROADMAP phase 0, final item).

The human labels are on a 1-5 scale; the LLM judge emits an expectation over the digits
1-6 (a continuous score on the 1-6 scale). Two things have to be handled:

1. **Scale alignment.** The human scale has 5 points and the judge scale 6. The judge is
   mapped onto the human scale by the affine map that sends the judge's 1->1 and 6->5
   (i.e. `1 + (judge - 1) * 4/5`), which is the standard way to compare ratings that use
   the same ordered labels with a different number of points. Both the raw and the mapped
   numbers are reported, and the correlations are also reported on the raw scale, since
   an affine map does not change Spearman and only rescales Pearson/MAE.

2. **Binning.** Rounding the judge score to the nearest integer makes exact agreement and
   Cohen's kappa well defined. The judge score is binned with `np.floor(x + 0.5)`, then
   clipped to the judge's 1-6 range and mapped to the human scale.

Inputs:
    human_annotation_100.jsonl                (repo root)
    results/full_run/judge_human_100.jsonl    (Stage 9 output)

Outputs:
    results/full_run/judge_human_validation.csv    per-row table
    results/full_run/judge_human_validation.md     the summary table for the paper
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
HUMAN_FILE = ROOT / "human_annotation_100.jsonl"
JUDGE_FILE = ROOT / "results/full_run/judge_human_100.jsonl"
OUT_CSV = ROOT / "results/full_run/judge_human_validation.csv"
OUT_MD = ROOT / "results/full_run/judge_human_validation.md"

HUMAN_MIN, HUMAN_MAX = 1, 5
JUDGE_MIN, JUDGE_MAX = 1, 6


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_human_scale(judge):
    """Affine map of the judge scale (1-6) onto the human scale (1-5)."""
    return HUMAN_MIN + (np.asarray(judge, dtype=float) - JUDGE_MIN) * (HUMAN_MAX - HUMAN_MIN) / (JUDGE_MAX - JUDGE_MIN)


def bin_to_int(scores, lo, hi):
    """Round to the nearest integer and clip into [lo, hi]."""
    return np.clip(np.floor(np.asarray(scores, dtype=float) + 0.5), lo, hi).astype(int)


human_rows = load_jsonl(HUMAN_FILE)
judge_rows = load_jsonl(JUDGE_FILE)
print(f"human rows: {len(human_rows)} | judge rows: {len(judge_rows)}")

human = pd.DataFrame(human_rows)[["id", "instruction", "response", "human_complexity",
                                   "human_quality", "reason"]]
judge_cols = [c for c in ("id", "llm_complexity", "llm_quality", "llama_usefulness",
                           "llama_usefulness_mass", "source") if c in judge_rows[0]]
judge = pd.DataFrame(judge_rows)[judge_cols]

df = human.merge(judge, on="id", how="inner").sort_values("id").reset_index(drop=True)
print(f"matched rows: {len(df)}")
if len(df) != len(human):
    print(f"WARNING: {len(human) - len(df)} human rows have no judge score")

# --- add the mapped and binned columns -------------------------------------------------------
for metric in ("complexity", "quality"):
    df[f"llm_{metric}_human_scale"] = to_human_scale(df[f"llm_{metric}"])
    df[f"llm_{metric}_binned"] = bin_to_int(to_human_scale(df[f"llm_{metric}"]), HUMAN_MIN, HUMAN_MAX)

results = {}
for metric in ("complexity", "quality"):
    h = df[f"human_{metric}"].astype(float).to_numpy()
    j_raw = df[f"llm_{metric}"].astype(float).to_numpy()
    j_scaled = df[f"llm_{metric}_human_scale"].to_numpy()
    j_binned = df[f"llm_{metric}_binned"].astype(int)

    sp_raw = spearmanr(h, j_raw)
    pe_raw = pearsonr(h, j_raw)
    kt = kendalltau(h, j_binned)

    results[metric] = {
        "n": len(h),
        "spearman": round(float(sp_raw.statistic), 4),
        "spearman_p": round(float(sp_raw.pvalue), 4),
        "pearson": round(float(pe_raw.statistic), 4),
        "pearson_p": round(float(pe_raw.pvalue), 4),
        "kendall_tau": round(float(kt.statistic), 4),
        "kendall_p": round(float(kt.pvalue), 4),
        "mae_raw_scale": round(float(mean_absolute_error(h, j_raw)), 4),
        "mae_human_scale": round(float(mean_absolute_error(h, j_scaled)), 4),
        "exact_agreement": round(float((df[f"human_{metric}"] == j_binned).mean()), 4),
        "within_1": round(float((np.abs(df[f"human_{metric}"].to_numpy() - j_binned) <= 1).mean()), 4),
        "kappa_quadratic": round(float(cohen_kappa_score(df[f"human_{metric}"], j_binned, weights="quadratic")), 4),
        "kappa_linear": round(float(cohen_kappa_score(df[f"human_{metric}"], j_binned, weights="linear")), 4),
        "kappa_unweighted": round(float(cohen_kappa_score(df[f"human_{metric}"], j_binned)), 4),
        "human_mean": round(float(h.mean()), 4),
        "judge_mean_raw": round(float(j_raw.mean()), 4),
        "judge_mean_human_scale": round(float(j_scaled.mean()), 4),
    }

# --- agreement by human score --------------------------------------------------------------
calib = {}
for metric in ("complexity", "quality"):
    g = df.groupby(f"human_{metric}")[f"llm_{metric}_human_scale"]
    calib[metric] = {
        str(k): {
            "n": int(v.count()),
            "judge_mean_human_scale": round(float(v.mean()), 3),
            "judge_sd": round(float(v.std()), 3) if v.count() > 1 else 0.0,
        }
        for k, v in g
    }

df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# --- markdown summary ------------------------------------------------------------------------
md = []
md.append("# Judge vs human validation (100 annotated rows)\n")
md.append(f"Rows: {len(df)} (Stage 9, Qwen2.5-7B-Instruct judge, the same rubric as Stage 3).\n")
md.append("The judge scores on a 1-6 scale; the human labels use 1-5. The judge is mapped onto the "
          "human scale with `1 + (judge - 1) * 4/5` and binned to the nearest integer for the "
          "agreement statistics. Spearman is unaffected by the map.\n")
md.append("| Metric | n | Spearman | p | Kendall τ | Pearson | MAE (1-5 scale) | Exact | ±1 | κ (quad) |")
md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for metric, label in (("complexity", "Complexity"), ("quality", "Quality")):
    r = results[metric]
    md.append(
        f"| {label} | {r['n']} | {r['spearman']:.3f} | {r['spearman_p']:.4f} | {r['kendall_tau']:.3f} | "
        f"{r['pearson']:.3f} | {r['mae_human_scale']:.3f} | {r['exact_agreement']:.3f} | "
        f"{r['within_1']:.3f} | {r['kappa_quadratic']:.3f} |"
    )
md.append("")
md.append("## Mean judge score by human score\n")
md.append("| Human score | n (complexity) | judge mean | n (quality) | judge mean |")
md.append("| --- | --- | --- | --- | --- |")
for level in sorted(set(df["human_complexity"]) | set(df["human_quality"])):
    c = calib["complexity"].get(str(level))
    q = calib["quality"].get(str(level))
    c_txt = f"{c['n']} | {c['judge_mean_human_scale']:.2f}" if c else "- | -"
    q_txt = f"{q['n']} | {q['judge_mean_human_scale']:.2f}" if q else "- | -"
    md.append(f"| {level} | {c_txt} | {q_txt} |")
md.append("")
md.append("## Score distributions\n")
md.append("| Scale | 1 | 2 | 3 | 4 | 5 | 6 |")
md.append("| --- | --- | --- | --- | --- | --- | --- |")
for name, series in (
    ("human complexity", df["human_complexity"]),
    ("human quality", df["human_quality"]),
    ("judge complexity (binned)", df["llm_complexity_binned"]),
    ("judge quality (binned)", df["llm_quality_binned"]),
):
    counts = series.value_counts().reindex(range(1, 7), fill_value=0)
    md.append(f"| {name} | " + " | ".join(str(int(v)) for v in counts) + " |")
md.append("")
md.append("## Score source\n")
md.append("| Source | Rows |")
md.append("| --- | --- |")
for src, n in df["source"].value_counts().items():
    md.append(f"| {src} | {n} |")

OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

print("\n" + "=" * 60)
print("JUDGE vs HUMAN VALIDATION")
print("=" * 60)
for metric, label in (("complexity", "COMPLEXITY"), ("quality", "QUALITY")):
    r = results[metric]
    print(f"\n{label} (n={r['n']})")
    print("-" * 40)
    for key in ("spearman", "spearman_p", "kendall_tau", "pearson", "mae_human_scale",
                "exact_agreement", "within_1", "kappa_quadratic", "kappa_linear",
                "human_mean", "judge_mean_human_scale"):
        print(f"{key:24s} {r[key]}")

print("\nwrote:")
print(f"  {OUT_CSV}")
print(f"  {OUT_MD}")

# stash the numbers for the master table
(ROOT / "results/full_run/judge_human_validation_summary.json").write_text(
    json.dumps({"results": results, "calibration": calib}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
