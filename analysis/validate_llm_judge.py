import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, cohen_kappa_score


HUMAN_FILE = "data/pilot/human_annotation_100.jsonl"
LLM_FILE = "scoring/results/llm_judge_100.jsonl"


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


# Load data
human = load_jsonl(HUMAN_FILE)
llm = load_jsonl(LLM_FILE)


# Convert to DataFrame
human_df = pd.DataFrame(human)
llm_df = pd.DataFrame(llm)


# Select required columns
human_df = human_df[
    ["id", "human_complexity", "human_quality"]
]

llm_df = llm_df[
    ["id", "llm_complexity", "llm_quality"]
]


# Match human and LLM annotations using ID
df = human_df.merge(
    llm_df,
    on="id",
    how="inner"
)


print("=" * 60)
print("HUMAN vs LLM JUDGE VALIDATION")
print("=" * 60)

print(f"Human rows:   {len(human_df)}")
print(f"LLM rows:     {len(llm_df)}")
print(f"Matched rows: {len(df)}")


print()
print("=" * 60)
print("BASIC AGREEMENT")
print("=" * 60)


# -------------------------
# Complexity agreement
# -------------------------

complexity_exact = (
    df["human_complexity"] ==
    df["llm_complexity"]
).mean()

complexity_within_1 = (
    (
        df["human_complexity"] -
        df["llm_complexity"]
    ).abs() <= 1
).mean()


# -------------------------
# Quality agreement
# -------------------------

quality_exact = (
    df["human_quality"] ==
    df["llm_quality"]
).mean()

quality_within_1 = (
    (
        df["human_quality"] -
        df["llm_quality"]
    ).abs() <= 1
).mean()


print(f"Complexity exact agreement: {complexity_exact:.3f}")
print(f"Complexity ±1 agreement:    {complexity_within_1:.3f}")

print()

print(f"Quality exact agreement:    {quality_exact:.3f}")
print(f"Quality ±1 agreement:       {quality_within_1:.3f}")


# -------------------------
# Score distributions
# -------------------------

print()
print("=" * 60)
print("HUMAN SCORE DISTRIBUTION")
print("=" * 60)

print("\nComplexity:")
print(
    df["human_complexity"]
    .value_counts()
    .sort_index()
)

print("\nQuality:")
print(
    df["human_quality"]
    .value_counts()
    .sort_index()
)


print()
print("=" * 60)
print("LLM SCORE DISTRIBUTION")
print("=" * 60)

print("\nComplexity:")
print(
    df["llm_complexity"]
    .value_counts()
    .sort_index()
)

print("\nQuality:")
print(
    df["llm_quality"]
    .value_counts()
    .sort_index()
)


# -------------------------
# Statistical validation
# -------------------------

print()
print("=" * 60)
print("STATISTICAL VALIDATION")
print("=" * 60)


# =========================
# Complexity
# =========================

complexity_spearman, complexity_spearman_p = spearmanr(
    df["human_complexity"],
    df["llm_complexity"]
)

complexity_pearson, complexity_pearson_p = pearsonr(
    df["human_complexity"],
    df["llm_complexity"]
)

complexity_mae = mean_absolute_error(
    df["human_complexity"],
    df["llm_complexity"]
)

complexity_kappa = cohen_kappa_score(
    df["human_complexity"],
    df["llm_complexity"],
    weights="quadratic"
)


# -------------------------
# Cross-tabulation analysis
# -------------------------

print()
print("=" * 60)
print("COMPLEXITY CONFUSION MATRIX")
print("=" * 60)

print(
    pd.crosstab(
        df["human_complexity"],
        df["llm_complexity"],
        rownames=["Human"],
        colnames=["LLM"],
        dropna=False
    )
)


print()
print("=" * 60)
print("QUALITY CONFUSION MATRIX")
print("=" * 60)

print(
    pd.crosstab(
        df["human_quality"],
        df["llm_quality"],
        rownames=["Human"],
        colnames=["LLM"],
        dropna=False
    )
)


print()
print("=" * 60)
print("MEAN LLM SCORE BY HUMAN SCORE")
print("=" * 60)

print("\nComplexity:")
print(
    df.groupby("human_complexity")["llm_complexity"]
    .mean()
    .round(3)
)

print("\nQuality:")
print(
    df.groupby("human_quality")["llm_quality"]
    .mean()
    .round(3)
)


print()
print("=" * 60)
print("MEAN HUMAN SCORE BY LLM SCORE")
print("=" * 60)

print("\nComplexity:")
print(
    df.groupby("llm_complexity")["human_complexity"]
    .mean()
    .round(3)
)

print("\nQuality:")
print(
    df.groupby("llm_quality")["human_quality"]
    .mean()
    .round(3)
)

# =========================
# Quality
# =========================

quality_spearman, quality_spearman_p = spearmanr(
    df["human_quality"],
    df["llm_quality"]
)

quality_pearson, quality_pearson_p = pearsonr(
    df["human_quality"],
    df["llm_quality"]
)

quality_mae = mean_absolute_error(
    df["human_quality"],
    df["llm_quality"]
)

quality_kappa = cohen_kappa_score(
    df["human_quality"],
    df["llm_quality"],
    weights="quadratic"
)


# =========================
# Print results
# =========================

print("\nCOMPLEXITY")
print("-" * 40)

print(f"Spearman correlation: {complexity_spearman:.4f}")
print(f"Spearman p-value:     {complexity_spearman_p:.4f}")
print(f"Pearson correlation:  {complexity_pearson:.4f}")
print(f"Pearson p-value:      {complexity_pearson_p:.4f}")
print(f"MAE:                  {complexity_mae:.4f}")
print(f"Weighted Kappa:       {complexity_kappa:.4f}")


print("\nQUALITY")
print("-" * 40)

print(f"Spearman correlation: {quality_spearman:.4f}")
print(f"Spearman p-value:     {quality_spearman_p:.4f}")
print(f"Pearson correlation:  {quality_pearson:.4f}")
print(f"Pearson p-value:      {quality_pearson_p:.4f}")
print(f"MAE:                  {quality_mae:.4f}")
print(f"Weighted Kappa:       {quality_kappa:.4f}")


# -------------------------
# Save matched dataset
# -------------------------

output_file = "analysis/human_vs_llm_100.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 60)
print(f"Saved: {output_file}")
print("=" * 60)