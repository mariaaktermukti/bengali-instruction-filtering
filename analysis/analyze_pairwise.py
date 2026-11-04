"""Stage 8 pairwise evaluation analysis and result promotion.

Computes:
1. Full 7x7 win matrix CSV
2. Pairwise comparison vs Arm B (random 10K) with 95% bootstrap CIs
3. Head-to-head comparison across all 21 pairs with 95% bootstrap CIs
4. Win rates vs Arm B broken down by domain
5. Promotes pairwise_summary.json to results/full_run/pairwise/
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

SRC_DIR = Path("outputs/full/s08_02/bangla_filtering/full/pairwise")
OUT_DIR = Path("results/full_run/pairwise")
MODELS = ["base", "A", "B", "C", "D", "E", "F"]
N_BOOTSTRAP = 2000
SEED = 42

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy pairwise_summary.json
    shutil.copy2(SRC_DIR / "pairwise_summary.json", OUT_DIR / "pairwise_summary.json")
    print(f"Copied pairwise_summary.json to {OUT_DIR}")
    
    # 2. Load all comparisons
    comparisons_file = SRC_DIR / "comparisons.jsonl"
    print(f"Loading {comparisons_file}...")
    records = []
    with open(comparisons_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} comparison records.")

    # Group by prompt id and pair (model_a, model_b)
    # pair_data[(pid, ma, mb)][order] = row
    prompt_pairs = defaultdict(dict)
    prompt_domains = {}
    for r in records:
        pid = r["id"]
        prompt_domains[pid] = r["domain"]
        ma, mb, order = r["model_a"], r["model_b"], r["order"]
        # order: 'ab' means response_a is ma, response_b is mb
        # 'ba' means response_a is mb, response_b is ma
        prompt_pairs[(pid, ma, mb)][order] = r

    prompt_ids = sorted(list(prompt_domains.keys()))
    n_prompts = len(prompt_ids)
    print(f"Total unique prompts: {n_prompts}")

    # Precompute per-prompt position-debiased win score for all ordered pairs (ma, mb)
    # win_score[(ma, mb)][pid] in [0.0, 1.0] where 1.0 means ma clearly won
    prompt_win = defaultdict(dict)
    for (pid, ma, mb), orders in prompt_pairs.items():
        ab_row = orders.get("ab")
        ba_row = orders.get("ba")
        if not ab_row or not ba_row:
            continue
        ab_v = ab_row["verdict_expected"]
        ba_v = ba_row["verdict_expected"]
        # In ab: 1=ma wins, 2=ma loses
        # In ba: 1=mb wins (ma loses), 2=ma wins
        score_ab = 1.0 if ab_v < 1.5 else (0.0 if ab_v > 2.5 else 0.5)
        score_ba = 1.0 if ba_v > 1.5 else (0.0 if ba_v < 1.5 else 0.5)
        ma_win = 0.5 * score_ab + 0.5 * score_ba
        prompt_win[(ma, mb)][pid] = ma_win
        prompt_win[(mb, ma)][pid] = 1.0 - ma_win

    # 3. Build Win Matrix
    win_matrix = {}
    for ma in MODELS:
        win_matrix[ma] = {}
        for mb in MODELS:
            if ma == mb:
                win_matrix[ma][mb] = 0.5000
            else:
                scores = [prompt_win[(ma, mb)][pid] for pid in prompt_ids if pid in prompt_win[(ma, mb)]]
                win_matrix[ma][mb] = round(float(np.mean(scores)), 4)
    
    df_win_matrix = pd.DataFrame(win_matrix).T
    df_win_matrix.to_csv(OUT_DIR / "pairwise_win_matrix.csv", index_label="model")
    print(f"Saved {OUT_DIR / 'pairwise_win_matrix.csv'}")

    # 4. Bootstrap CIs
    rng = np.random.default_rng(SEED)
    boot_indices = rng.integers(0, n_prompts, size=(N_BOOTSTRAP, n_prompts))

    # Bootstrap vs B
    rows_vs_b = []
    for m in MODELS:
        if m == "B":
            continue
        scores = np.array([prompt_win[(m, "B")][pid] for pid in prompt_ids])
        mean_val = float(np.mean(scores))
        boot_means = np.mean(scores[boot_indices], axis=1)
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        p_higher = float(np.mean(boot_means > 0.50))
        rows_vs_b.append({
            "model": m,
            "vs": "B",
            "n_prompts": n_prompts,
            "win_rate": round(mean_val, 4),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "p_higher_than_50pct": round(p_higher, 3)
        })
    df_vs_b = pd.DataFrame(rows_vs_b)
    df_vs_b.to_csv(OUT_DIR / "pairwise_vs_B.csv", index=False)
    print(f"Saved {OUT_DIR / 'pairwise_vs_B.csv'}")

    # Bootstrap head-to-head for all 21 pairs
    rows_h2h = []
    import itertools
    for ma, mb in itertools.combinations(MODELS, 2):
        scores = np.array([prompt_win[(ma, mb)][pid] for pid in prompt_ids])
        mean_val = float(np.mean(scores))
        boot_means = np.mean(scores[boot_indices], axis=1)
        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))
        p_higher = float(np.mean(boot_means > 0.50))
        rows_h2h.append({
            "model_A": ma,
            "model_B": mb,
            "n_prompts": n_prompts,
            "win_rate_A": round(mean_val, 4),
            "win_rate_B": round(1.0 - mean_val, 4),
            "ci95_low_A": round(ci_low, 4),
            "ci95_high_A": round(ci_high, 4),
            "p_A_beats_B": round(p_higher, 3)
        })
    df_h2h = pd.DataFrame(rows_h2h)
    df_h2h.to_csv(OUT_DIR / "pairwise_head_to_head.csv", index=False)
    print(f"Saved {OUT_DIR / 'pairwise_head_to_head.csv'}")

    # 5. Domain breakdown vs B
    domains = sorted(list(set(prompt_domains.values())))
    domain_rows = []
    for d in domains:
        d_pids = [pid for pid in prompt_ids if prompt_domains[pid] == d]
        n_d = len(d_pids)
        row = {"domain": d, "n_prompts": n_d}
        for m in MODELS:
            if m == "B":
                continue
            scores = [prompt_win[(m, "B")][pid] for pid in d_pids]
            row[f"{m}_win_vs_B"] = round(float(np.mean(scores)), 4) if scores else None
        domain_rows.append(row)
    df_domain = pd.DataFrame(domain_rows)
    df_domain.to_csv(OUT_DIR / "pairwise_by_domain.csv", index=False)
    print(f"Saved {OUT_DIR / 'pairwise_by_domain.csv'}")

    print("\nSummary of Pairwise Win Rates vs B:")
    print(df_vs_b.to_string(index=False))

    print("\nSelected Key Head-to-Head Comparisons:")
    key_pairs = [("C", "E"), ("D", "E"), ("D", "F"), ("C", "F"), ("C", "D"), ("A", "B"), ("A", "D")]
    for ma, mb in key_pairs:
        subset = df_h2h[(df_h2h["model_A"] == ma) & (df_h2h["model_B"] == mb)]
        if len(subset) == 0:
            subset = df_h2h[(df_h2h["model_A"] == mb) & (df_h2h["model_B"] == ma)]
        print(subset.to_string(index=False))

if __name__ == "__main__":
    main()

