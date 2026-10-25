import glob
import json
import math
import os


RESULTS_DIR = "results/lora_arms"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "summary.csv")

COLUMNS = [
    "arm",
    "train_examples",
    "truncated",
    "steps",
    "train_loss",
    "heldout_loss_before",
    "heldout_loss_after",
    "heldout_loss_change",
    "heldout_ppl_before",
    "heldout_ppl_after",
    "training_minutes",
    "peak_gpu_gib"
]


def main():

    rows = []

    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "arm_*_training_report.json"))):

        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)

        before = report["heldout_loss_before"]
        after = report["heldout_loss_after"]

        rows.append({
            "arm": report["arm"],
            "train_examples": report["token_stats"]["examples_used"],
            "truncated": report["token_stats"]["truncated_to_max_length"],
            "steps": report["global_steps"],
            "train_loss": round(report["train_loss"], 4),
            "heldout_loss_before": round(before, 4),
            "heldout_loss_after": round(after, 4),
            "heldout_loss_change": round(after - before, 4),
            "heldout_ppl_before": round(math.exp(before), 3),
            "heldout_ppl_after": round(math.exp(after), 3),
            "training_minutes": report["training_minutes"],
            "peak_gpu_gib": report["peak_gpu_gib"]
        })

    if not rows:
        print("No reports found in", RESULTS_DIR)
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        f.write(",".join(COLUMNS) + "\n")

        for row in rows:
            f.write(",".join(str(row[c]) for c in COLUMNS) + "\n")

    print(f"{'arm':<4} {'n':>4} {'steps':>5} {'held-out before':>16} {'after':>8} {'change':>8} {'min':>6}")

    for row in sorted(rows, key=lambda r: r["heldout_loss_after"]):
        print(
            f"{row['arm']:<4} {row['train_examples']:>4} {row['steps']:>5} "
            f"{row['heldout_loss_before']:>16.4f} {row['heldout_loss_after']:>8.4f} "
            f"{row['heldout_loss_change']:>8.4f} {row['training_minutes']:>6}"
        )

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
