"""Kaggle CPU wrapper for the full-run held-out bootstrap evaluation."""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


N_BOOTSTRAP = 10_000
SEED = 42
OUT = "/kaggle/working/full_run_results"


def load_runs(root):
    runs, base = {}, None
    base_path = os.path.join(root, "base_heldout_per_example.csv")
    if os.path.exists(base_path):
        base = pd.read_csv(base_path)
    for path in sorted(glob.glob(os.path.join(root, "arm_*_seed*", "heldout_per_example.csv"))):
        run_dir = os.path.dirname(path)
        name = os.path.basename(run_dir)
        with open(os.path.join(run_dir, "training_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        if report["complete"]:
            runs[name] = (pd.read_csv(path), report["token_stats"]["total_tokens"])
    if base is None or not runs:
        raise SystemExit("No complete Stage 5 outputs found.")
    ids = set(base["id"])
    for name, (df, _) in runs.items():
        if len(df) != len(ids) or set(df["id"]) != ids or df["loss"].isna().any():
            raise SystemExit(f"{name}: held-out rows do not match base.")
    return runs, base


def bootstrap_table(pivot, tokens):
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(pivot), size=(N_BOOTSTRAP, len(pivot)))
    weights = tokens.to_numpy(dtype=float)
    rows = []
    for model in pivot.columns:
        values = pivot[model].to_numpy()
        row = {
            "model": model,
            "n": len(values),
            "token_weighted_loss": round(float((values * weights).sum() / weights.sum()), 4),
        }
        for ref in ("base", "B"):
            if model == ref:
                continue
            diff = values - pivot[ref].to_numpy()
            boot = (diff[samples] * weights[samples]).sum(axis=1) / weights[samples].sum(axis=1)
            row[f"diff_vs_{ref}"] = round(float((diff * weights).sum() / weights.sum()), 4)
            row[f"ci95_low_vs_{ref}"] = round(float(np.quantile(boot, 0.025)), 4)
            row[f"ci95_high_vs_{ref}"] = round(float(np.quantile(boot, 0.975)), 4)
            row[f"p_lower_vs_{ref}"] = round(float((boot < 0).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    inputs = glob.glob("/kaggle/input/**/full/lora", recursive=True)
    if not inputs:
        raise SystemExit("Attach the completed Stage 5 kernel as an input.")
    runs, base = load_runs(inputs[0])
    os.makedirs(OUT, exist_ok=True)

    frames = [base.assign(model="base")]
    per_seed = []
    for name, (df, train_tokens) in runs.items():
        arm = name.split("_")[1]
        per_seed.append({
            "run": name,
            "mean_loss": round(float(df["loss"].mean()), 4),
            "token_weighted_loss": round(float((df["loss"] * df["response_tokens"]).sum()
                                               / df["response_tokens"].sum()), 4),
            "train_tokens": train_tokens,
        })
        frames.append(df.assign(model=arm))

    merged = (pd.concat(frames)
              .groupby(["model", "id", "domain", "response_tokens"], as_index=False)["loss"]
              .mean().round({"loss": 6}))
    merged[["model", "id", "domain", "loss", "response_tokens"]].to_csv(
        os.path.join(OUT, "heldout_per_example.csv"), index=False)

    ids = sorted(set(base["id"]))
    losses = defaultdict(dict)
    domains = {}
    for row in merged.itertuples():
        losses[row.model][row.id] = row.loss
        domains[row.id] = row.domain
    vectors = {model: np.array([losses[model][item] for item in ids]) for model in losses}
    rng = np.random.default_rng(SEED)
    comparison = []
    for model, values in vectors.items():
        row = {"model": model, "n": len(values), "mean_loss": round(float(values.mean()), 4)}
        for ref in ("base", "B"):
            if model == ref:
                continue
            diff = values - vectors[ref]
            samples = rng.integers(0, len(diff), size=(N_BOOTSTRAP, len(diff)))
            boot = diff[samples].mean(axis=1)
            row[f"diff_vs_{ref}"] = round(float(diff.mean()), 4)
            row[f"ci95_low_vs_{ref}"] = round(float(np.quantile(boot, 0.025)), 4)
            row[f"ci95_high_vs_{ref}"] = round(float(np.quantile(boot, 0.975)), 4)
            row[f"p_lower_vs_{ref}"] = round(float((boot < 0).mean()), 3)
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(os.path.join(OUT, "heldout_comparison.csv"), index=False)

    domain_rows = []
    for model, values in vectors.items():
        for domain in sorted(set(domains.values())):
            idx = [i for i, item in enumerate(ids) if domains[item] == domain]
            domain_rows.append({"model": model, "domain": domain, "n": len(idx),
                                "mean_loss": round(float(values[idx].mean()), 4)})
    pd.DataFrame(domain_rows).to_csv(os.path.join(OUT, "heldout_by_domain.csv"), index=False)

    tokens = base.set_index("id")["response_tokens"].reindex(ids)
    bootstrap_table(pd.DataFrame(vectors, index=ids), tokens).to_csv(
        os.path.join(OUT, "token_weighted.csv"), index=False)
    pd.DataFrame(per_seed).to_csv(os.path.join(OUT, "per_seed.csv"), index=False)
    print(pd.DataFrame(comparison).to_string(index=False))
    print(bootstrap_table(pd.DataFrame(vectors, index=ids), tokens).to_string(index=False))


if __name__ == "__main__":
    main()
