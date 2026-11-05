"""Match the 100 human-annotated rows against the full-run pool (arm A) and the production judge scores.

The human labels were made on pilot rows (ids 0-99). The full run sampled a different
100K+500 subset of the same source dataset, so only some of the annotated rows are in
the clean pool. For those, the production Stage 3 judge scores are already available
(arm_A.jsonl carries id / c / q for every pool row), which gives an immediate
judge-vs-human comparison without any GPU work.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "human_annotation_100.jsonl"
ARM_A = ROOT / "outputs/full/s04_v4/bangla_filtering/full/arms/arm_A.jsonl"
HELD_OUT = ROOT / "outputs/full/s04_v4/bangla_filtering/full/arms/heldout.jsonl"
OUT_CSV = ROOT / "results/full_run/judge_human_overlap_pool.csv"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_arm_jsonl(path):
    keep = ["id", "instruction", "output", "domain", "c", "q"]
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows.append({k: r.get(k) for k in keep})
    return pd.DataFrame(rows)


human = load_jsonl(HUMAN)
pool = load_arm_jsonl(ARM_A)
held = load_arm_jsonl(HELD_OUT)
print(f"human rows: {len(human)} | pool rows: {len(pool)} | held-out rows: {len(held)}")

for frame, name in ((pool, "pool"), (held, "held-out")):
    frame["instruction"] = frame["instruction"].astype(str).str.strip()
    frame["output"] = frame["output"].astype(str).str.strip()

lookup = {}
for source, frame in (("pool", pool), ("heldout", held)):
    for rec in frame.to_dict("records"):
        lookup.setdefault((rec["instruction"], rec["output"]), (source, rec))

rows = []
for h in human:
    key = (h["instruction"].strip(), h["response"].strip())
    hit = lookup.get(key)
    rec = {
        "human_id": h["id"],
        "human_complexity": h["human_complexity"],
        "human_quality": h["human_quality"],
        "in_pool": hit is not None and hit[0] == "pool",
        "in_heldout": hit is not None and hit[0] == "heldout",
        "pool_id": hit[1]["id"] if hit else None,
        "domain": hit[1]["domain"] if hit else None,
        "llm_complexity": hit[1]["c"] if hit else None,
        "llm_quality": hit[1]["q"] if hit else None,
    }
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nwrote {OUT_CSV}")
print(f"in pool: {int(df['in_pool'].sum())} | in held-out: {int(df['in_heldout'].sum())} | not in run: {int((~df['in_pool'] & ~df['in_heldout']).sum())}")

matched = df[df["in_pool"]]
if len(matched):
    from scipy.stats import pearsonr, spearmanr

    for metric in ("complexity", "quality"):
        h = matched[f"human_{metric}"].astype(float)
        l = matched[f"llm_{metric}"].astype(float)
        print(f"\n{metric}: n={len(matched)} spearman={spearmanr(h, l).statistic:.4f} "
              f"pearson={pearsonr(h, l).statistic:.4f} mae_raw={abs(h - l).mean():.4f}")
