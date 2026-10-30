"""Full run, Stage 6 (CPU, runs locally): compare the arms on the held-out losses that Stage 5 wrote.

    python kaggle/full_run/s06_eval/s06_bootstrap.py outputs/full/s05_NN [outputs/full/s05_MM ...]

Later sessions carry over earlier runs, so the latest session's output is normally enough. Runs found twice count once.
An arm with several seeds is averaged per example, then scripts/compare_arms_bootstrap.py runs on one model per arm.
The bootstrap resamples held-out examples only, so seed variance shows up in per_seed.csv, not in the intervals.
Outputs in results/full_run/:
    heldout_per_example.csv   one row per arm x example (seeds averaged)
    heldout_comparison.csv    mean per-example loss, paired bootstrap vs base and vs B
    heldout_by_domain.csv     mean per-example loss by domain
    token_weighted.csv        the same comparison with each example weighted by its response tokens
    per_seed.csv              both means per run, with training tokens
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(REPO, "results", "full_run")
REFERENCES = ("base", "B")
N_BOOTSTRAP = 10_000
SEED = 42


def load_runs(session_dirs):
    runs, base = {}, None
    for session in session_dirs:
        root = os.path.join(session, "bangla_filtering", "full", "lora")
        if base is None and os.path.exists(os.path.join(root, "base_heldout_per_example.csv")):
            base = pd.read_csv(os.path.join(root, "base_heldout_per_example.csv"))
        for path in sorted(glob.glob(os.path.join(root, "arm_*_seed*", "heldout_per_example.csv"))):
            name = os.path.basename(os.path.dirname(path))
            report = os.path.join(os.path.dirname(path), "training_report.json")
            if name not in runs and json.load(open(report, encoding="utf-8"))["complete"]:
                runs[name] = (pd.read_csv(path), json.load(open(report, encoding="utf-8"))["token_stats"]["total_tokens"])
    if base is None or not runs:
        raise SystemExit("no Stage 5 output found under " + ", ".join(session_dirs))
    ids = set(base["id"])
    for name, (df, _) in runs.items():
        if len(df) != len(ids) or set(df["id"]) != ids or df["loss"].isna().any():
            raise SystemExit(f"{name}: held-out rows do not match the base model's")
    return runs, base


def weighted_table(pivot, tokens):
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(pivot), size=(N_BOOTSTRAP, len(pivot)))
    w = tokens.to_numpy(dtype=float)
    weighted = lambda v: (v[samples] * w[samples]).sum(axis=1) / w[samples].sum(axis=1)
    rows = []
    for model in pivot.columns:
        v = pivot[model].to_numpy()
        row = {"model": model, "n": len(v), "token_weighted_loss": round(float((v * w).sum() / w.sum()), 4)}
        for ref in REFERENCES:
            if model != ref:
                diff = v - pivot[ref].to_numpy()
                boot = weighted(diff)
                row[f"diff_vs_{ref}"] = round(float((diff * w).sum() / w.sum()), 4)
                row[f"ci95_low_vs_{ref}"] = round(float(np.quantile(boot, 0.025)), 4)
                row[f"ci95_high_vs_{ref}"] = round(float(np.quantile(boot, 0.975)), 4)
                row[f"p_lower_vs_{ref}"] = round(float((boot < 0).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    runs, base = load_runs(sys.argv[1:])
    os.makedirs(OUT, exist_ok=True)

    tokens = base.set_index("id")["response_tokens"]
    per_seed, frames = [], [base.assign(model="base")]
    for name, (df, train_tokens) in runs.items():
        per_seed.append({"run": name, "mean_loss": round(df["loss"].mean(), 4),
                         "token_weighted_loss": round((df["loss"] * df["response_tokens"]).sum() / df["response_tokens"].sum(), 4),
                         "train_tokens": train_tokens})
        frames.append(df.assign(model=re.match(r"arm_(\w+)_seed", name).group(1)))
    pd.DataFrame(per_seed).to_csv(os.path.join(OUT, "per_seed.csv"), index=False)

    merged = (pd.concat(frames).groupby(["model", "id", "domain", "response_tokens"], as_index=False)["loss"].mean()
              .round({"loss": 6}))
    merged_csv = os.path.join(OUT, "heldout_per_example.csv")
    merged[["model", "id", "domain", "loss", "response_tokens"]].to_csv(merged_csv, index=False)

    subprocess.check_call([sys.executable, os.path.join(REPO, "scripts", "compare_arms_bootstrap.py"), merged_csv, OUT])

    pivot = merged.pivot(index="id", columns="model", values="loss")
    table = weighted_table(pivot, tokens.reindex(pivot.index))
    table.to_csv(os.path.join(OUT, "token_weighted.csv"), index=False)
    print("\nToken-weighted:\n" + table.to_string(index=False))
    print("\nPer seed:\n" + pd.DataFrame(per_seed).to_string(index=False))


if __name__ == "__main__":
    main()
