"""Join the blind spot-check sheet with the key and turn it into arm-level statistics.

The sheet (`reports/spotcheck_blind.csv`, produced by Stage 4) mixes 50 rows from arm C
and 50 from arm B with the arm hidden. A rating of those rows by the *same* judge that
selected the arms would be circular: arm C was chosen to maximise the judge's complexity x
quality. So the sheet is rated by an independent model from a different family
(Llama-3.1-8B-Instruct, Stage 9), on a 1-5 scale that matches the human `rating_1to5`
column. The Qwen judge's own scores for the same items are reported next to it, so the
paper can show both the circular and the non-circular view.

This script:
  * joins the blind ratings with the key,
  * reports the mean rating per arm with a bootstrap CI,
  * tests the C-vs-B difference with a paired bootstrap over the items,
  * reports the Qwen judge's own complexity/quality for the same rows.

The human ratings file is optional: if it is absent the script still reports the
independent-rater and judge statistics and says that the human part is pending.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/full/s04_v4/bangla_filtering/full/reports"
BLIND = REPORTS / "spotcheck_blind.csv"
KEY = REPORTS / "spotcheck_key.csv"
RATINGS = ROOT / "results/full_run/spotcheck_ratings.csv"        # human ratings (to be filled)
JUDGE_SPOT = ROOT / "results/full_run/spotcheck_judge.csv"      # Stage 9 output (Qwen + Llama)
OUT_CSV = ROOT / "results/full_run/spotcheck_results.csv"
OUT_MD = ROOT / "results/full_run/spotcheck_results.md"

BOOT = 10_000
SEED = 42


def boot_ci(values, stat=np.mean, n_boot=BOOT, seed=SEED):
    """Bootstrap CI of a statistic over a 1-D sample."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    stats = stat(values[idx], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def two_sample_bootstrap(c_vals, b_vals):
    """Bootstrap of mean(C) - mean(B) for two independent groups.

    The blind sheet holds 50 *different* items per arm (each item comes from exactly one
    arm), so the arms cannot be paired item by item: this is an unpaired two-sample
    comparison and the bootstrap resamples each arm separately.
    """
    c_vals = np.asarray(c_vals, dtype=float)
    b_vals = np.asarray(b_vals, dtype=float)
    if len(c_vals) < 2 or len(b_vals) < 2:
        return None
    rng = np.random.default_rng(SEED)
    idx_c = rng.integers(0, len(c_vals), size=(BOOT, len(c_vals)))
    idx_b = rng.integers(0, len(b_vals), size=(BOOT, len(b_vals)))
    boot = c_vals[idx_c].mean(axis=1) - b_vals[idx_b].mean(axis=1)
    return {
        "n_C": len(c_vals),
        "n_B": len(b_vals),
        "mean_C": round(float(c_vals.mean()), 4),
        "mean_B": round(float(b_vals.mean()), 4),
        "diff_C_minus_B": round(float(c_vals.mean() - b_vals.mean()), 4),
        "diff_ci95_low": round(float(np.percentile(boot, 2.5)), 4),
        "diff_ci95_high": round(float(np.percentile(boot, 97.5)), 4),
        "p_C_better": round(float((boot > 0).mean()), 4),
    }


blind = pd.read_csv(BLIND)
key = pd.read_csv(KEY)
print(f"blind sheet: {len(blind)} rows | key: {len(key)} rows")

df = blind.merge(key, on="item", how="left", validate="one_to_one")
print("arm counts:", df["arm"].value_counts().to_dict())

# --- human ratings (optional) -----------------------------------------------------------------
df["human_rating"] = np.nan
df["human_notes"] = ""
if RATINGS.exists():
    rat = pd.read_csv(RATINGS)
    if "item" in rat.columns:
        rat = rat[["item", "rating", "notes"]].rename(columns={"rating": "human_rating",
                                                               "notes": "human_notes"})
        df = df.merge(rat, on="item", how="left")
        df["human_rating"] = pd.to_numeric(df["human_rating"], errors="coerce")
        print(f"human ratings loaded: {int(df['human_rating'].notna().sum())}/{len(df)}")
    else:
        print("ratings file has no 'item' column; ignoring it")
else:
    print(f"no ratings file at {RATINGS} (human part pending)")

# --- judge scores from Stage 9 ----------------------------------------------------------------
df["llm_complexity"] = np.nan
df["llm_quality"] = np.nan
df["llama_usefulness"] = np.nan
if JUDGE_SPOT.exists():
    js = pd.read_csv(JUDGE_SPOT)
    keep = [c for c in ("item", "llm_complexity", "llm_quality", "llama_usefulness") if c in js.columns]
    # the blind sheet already has empty llm_* / llama_* columns; drop them before merging
    df = df.drop(columns=[c for c in keep if c != "item" and c in df.columns], errors="ignore")
    df = df.merge(js[keep], on="item", how="left")
    print(f"judge scores joined: {int(df['llm_complexity'].notna().sum())}/{len(df)}")
else:
    print(f"no Stage 9 judge file at {JUDGE_SPOT}")

# --- per-arm summary ---------------------------------------------------------------------------
rows = []
for arm in sorted(df["arm"].unique()):
    sub = df[df["arm"] == arm]
    hr = sub["human_rating"].dropna()
    llama = sub["llama_usefulness"].dropna()
    hr_lo, hr_hi = boot_ci(hr) if len(hr) > 1 else (np.nan, np.nan)
    ll_lo, ll_hi = boot_ci(llama) if len(llama) > 1 else (np.nan, np.nan)
    rows.append({
        "arm": arm,
        "n_items": len(sub),
        "n_human_rated": len(hr),
        "human_mean_rating": round(float(hr.mean()), 4) if len(hr) else np.nan,
        "human_rating_ci95_low": round(hr_lo, 4) if len(hr) > 1 else np.nan,
        "human_rating_ci95_high": round(hr_hi, 4) if len(hr) > 1 else np.nan,
        "independent_mean_rating": round(float(llama.mean()), 4) if len(llama) else np.nan,
        "independent_ci95_low": round(ll_lo, 4) if len(llama) > 1 else np.nan,
        "independent_ci95_high": round(ll_hi, 4) if len(llama) > 1 else np.nan,
        "judge_mean_complexity": round(float(sub["llm_complexity"].mean()), 4),
        "judge_mean_quality": round(float(sub["llm_quality"].mean()), 4),
        "judge_mean_cq": round(float((sub["llm_complexity"] * sub["llm_quality"]).mean()), 4),
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print("\n" + summary.to_string(index=False))

# --- the independent rater: the non-circular arm comparison ------------------------------------
independent = {}
if {"C", "B"}.issubset(set(df["arm"].unique())):
    res = two_sample_bootstrap(
        df.loc[df["arm"] == "C", "llama_usefulness"].dropna().to_numpy(),
        df.loc[df["arm"] == "B", "llama_usefulness"].dropna().to_numpy(),
    )
    if res:
        independent = {"rater": "unsloth/Llama-3.1-8B-Instruct", **res}

# --- human ratings: the same comparison, if the sheet has been filled in ------------------------
human_cmp = {"has_ratings": bool(df["human_rating"].notna().any())}
if {"C", "B"}.issubset(set(df["arm"].unique())):
    res = two_sample_bootstrap(
        df.loc[df["arm"] == "C", "human_rating"].dropna().to_numpy(),
        df.loc[df["arm"] == "B", "human_rating"].dropna().to_numpy(),
    )
    if res:
        human_cmp.update(res)

# --- the judge's own arm-level view (no human needed) -------------------------------------------
judge_rows = {}
for arm in sorted(df["arm"].unique()):
    sub = df[df["arm"] == arm]
    c = sub["llm_complexity"].dropna()
    q = sub["llm_quality"].dropna()
    if len(c):
        judge_rows[arm] = {
            "n": int(len(c)),
            "mean_complexity": round(float(c.mean()), 4),
            "mean_quality": round(float(q.mean()), 4),
            "mean_cq": round(float((c * q).mean()), 4),
            "cq_ci95_low": round(boot_ci(c * q)[0], 4),
            "cq_ci95_high": round(boot_ci(c * q)[1], 4),
        }

# --- markdown -----------------------------------------------------------------------------------
md = []
md.append("# Blind spot-check: arm B (random 10K) vs arm C (DEITA, c x q + diversity)\n")
md.append(f"Items: {len(df)} (50 per arm, arm hidden in the sheet). "
          "Source: Stage 4 `reports/spotcheck_blind.csv`, key `reports/spotcheck_key.csv`.\n")

md.append("## Qwen judge (the model that selected the arms - circular for this comparison)\n")
if judge_rows:
    md.append("| Arm | n | mean complexity | mean quality | mean c x q | c x q 95% CI |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for arm, r in sorted(judge_rows.items()):
        md.append(f"| {arm} | {r['n']} | {r['mean_complexity']:.3f} | {r['mean_quality']:.3f} | "
                  f"{r['mean_cq']:.3f} | [{r['cq_ci95_low']:.3f}, {r['cq_ci95_high']:.3f}] |")
else:
    md.append("_Stage 9 judge scores not available yet._")
md.append("")

md.append("## Independent rater (non-circular)\n")
if independent:
    md.append(f"Rater: `{independent['rater']}` - a different model family from the Qwen judge that "
              "selected the arms, so this comparison does not reward the selection criterion itself. "
              f"Two independent samples: {independent['n_C']} C items vs {independent['n_B']} B items.\n")
    md.append("| Arm | n | mean rating | 95% CI |")
    md.append("| --- | --- | --- | --- |")
    for _, r in summary.sort_values("arm").iterrows():
        if pd.notna(r["independent_mean_rating"]):
            md.append(f"| {r['arm']} | {int(r['n_items'])} | {r['independent_mean_rating']:.3f} | "
                      f"[{r['independent_ci95_low']:.3f}, {r['independent_ci95_high']:.3f}] |")
    md.append("")
    md.append(f"Mean C - mean B = **{independent['diff_C_minus_B']:+.3f}** "
              f"(95% CI [{independent['diff_ci95_low']:+.3f}, {independent['diff_ci95_high']:+.3f}], "
              f"p(C better) = {independent['p_C_better']:.3f}).")
else:
    md.append("_Stage 9 independent-rater scores not available yet._")
md.append("")

md.append("## Human ratings (blind)\n")
if human_cmp.get("has_ratings"):
    md.append(f"Two independent samples: {human_cmp['n_C']} C items vs {human_cmp['n_B']} B items.\n")
    md.append("| Arm | n rated | mean rating | 95% CI |")
    md.append("| --- | --- | --- | --- |")
    for _, r in summary.sort_values("arm").iterrows():
        if r["n_human_rated"]:
            md.append(f"| {r['arm']} | {int(r['n_human_rated'])} | {r['human_mean_rating']:.3f} | "
                      f"[{r['human_rating_ci95_low']:.3f}, {r['human_rating_ci95_high']:.3f}] |")
    md.append("")
    md.append(f"Mean C - mean B = **{human_cmp['diff_C_minus_B']:+.3f}** "
              f"(95% CI [{human_cmp['diff_ci95_low']:+.3f}, {human_cmp['diff_ci95_high']:+.3f}], "
              f"p(C better) = {human_cmp['p_C_better']:.3f}).")
else:
    md.append("_No human ratings yet. Fill the `item` / `rating` / `notes` columns of "
              "`results/full_run/spotcheck_ratings.csv` and re-run this script._")
md.append("")

OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

print(f"\nwrote {OUT_CSV}")
print(f"wrote {OUT_MD}")

(ROOT / "results/full_run/spotcheck_summary.json").write_text(
    json.dumps({"summary": summary.to_dict(orient="records"),
                "human_vs_arm": human_cmp,
                "independent_rater": independent,
                "judge_by_arm": judge_rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
