"""Paired bootstrap comparison of held-out loss across arms.

    python scripts/compare_arms_bootstrap.py PER_EXAMPLE_CSV

Reads the output of heldout_per_example.py. For every model it reports the mean
held-out loss and, against the base model and against arm B (random selection),
the mean paired difference with a 95% bootstrap interval over held-out examples.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np


REFERENCE_ARM = "B"
N_BOOTSTRAP = 10_000
SEED = 42

OUTPUT_FILE = "results/lora_arms/heldout_comparison.csv"
DOMAIN_FILE = "results/lora_arms/heldout_by_domain.csv"


def load(path):

    losses = defaultdict(dict)
    domains = {}

    with open(path, "r", encoding="utf-8") as f:

        for row in csv.DictReader(f):
            losses[row["model"]][row["id"]] = float(row["loss"])
            domains[row["id"]] = row["domain"]

    return losses, domains


def paired_interval(a, b, rng):

    diff = a - b

    samples = rng.integers(0, len(diff), size=(N_BOOTSTRAP, len(diff)))
    means = diff[samples].mean(axis=1)

    return diff.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975), (means < 0).mean()


def main():

    losses, domains = load(sys.argv[1])

    ids = sorted(losses["base"])
    vectors = {m: np.array([losses[m][i] for i in ids]) for m in losses}

    rng = np.random.default_rng(SEED)

    rows = []

    for model, vector in vectors.items():

        row = {"model": model, "n": len(ids), "mean_loss": round(float(vector.mean()), 4)}

        for ref_name in ("base", REFERENCE_ARM):

            if model == ref_name or ref_name not in vectors:
                continue

            mean, low, high, p_better = paired_interval(vector, vectors[ref_name], rng)

            row[f"diff_vs_{ref_name}"] = round(float(mean), 4)
            row[f"ci95_low_vs_{ref_name}"] = round(float(low), 4)
            row[f"ci95_high_vs_{ref_name}"] = round(float(high), 4)
            row[f"p_lower_vs_{ref_name}"] = round(float(p_better), 3)

        rows.append(row)

    columns = ["model", "n", "mean_loss"]

    for ref_name in ("base", REFERENCE_ARM):
        columns += [f"diff_vs_{ref_name}", f"ci95_low_vs_{ref_name}",
                    f"ci95_high_vs_{ref_name}", f"p_lower_vs_{ref_name}"]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    for row in rows:
        print(row)

    # Per-domain mean loss, to see which kinds of task each arm helps
    domain_ids = defaultdict(list)

    for i, example_id in enumerate(ids):
        domain_ids[domains[example_id]].append(i)

    with open(DOMAIN_FILE, "w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["model", "domain", "n", "mean_loss"])

        for model, vector in vectors.items():
            for domain, idx in sorted(domain_ids.items(), key=lambda x: -len(x[1])):
                writer.writerow([model, domain, len(idx), round(float(vector[idx].mean()), 4)])

    print("\nSaved:", OUTPUT_FILE)
    print("Saved:", DOMAIN_FILE)


if __name__ == "__main__":
    main()
