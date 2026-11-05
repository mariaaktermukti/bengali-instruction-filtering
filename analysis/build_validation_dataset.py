"""Build the Kaggle validation dataset for Stage 9 (judge vs human + blind spot-check).

Contents (small, ~1 MB total):
    human_annotation_100.jsonl  the 100 human-annotated rows (copied verbatim)
    spotcheck_blind.csv         the 100 blind spot-check items (50 B + 50 C, arm hidden)
    pool_matches.csv            the pool rows whose instruction matches a human row, with
                                their production Stage 3 scores (c, q) and a loose response
                                hash, so the kernel can tell the 29 reusable rows from the
                                71 that need a fresh judge pass
    spotcheck_key.csv           the key for the blind sheet (not read by the kernel; used
                                locally to join the ratings)

Why a loose key: Stage 1 normalised the pool text (NFC, whitespace, punctuation), so the
pilot's raw text and the pool's text differ byte-wise but are the same example. Matching on
NFC + punctuation/whitespace-insensitive text recovers the 29 rows HANDOFF records.
"""

import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "outputs/validation_dataset"
REPORTS = ROOT / "outputs/full/s04_v4/bangla_filtering/full/reports"
ARM_A = ROOT / "outputs/full/s04_v4/bangla_filtering/full/arms/arm_A.jsonl"

PUNCT_WS = re.compile(r"[\s।॥!-/:-@\[-`{-~“”‘’…–—]+")


def loose(s):
    """Same key as the Stage 9 kernel: NFC + punctuation/whitespace-insensitive."""
    return PUNCT_WS.sub("", unicodedata.normalize("NFC", str(s)).lower())


def loose_hash(s):
    return hashlib.sha1(loose(s).encode("utf-8")).hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


BUILD.mkdir(parents=True, exist_ok=True)

human = load_jsonl(ROOT / "human_annotation_100.jsonl")
human_instr = {loose(h["instruction"]) for h in human}
print(f"human rows: {len(human)}, distinct loose instructions: {len(human_instr)}")

# 1. human annotations
shutil.copy2(ROOT / "human_annotation_100.jsonl", BUILD / "human_annotation_100.jsonl")

# 2. blind spot-check sheet + key
shutil.copy2(REPORTS / "spotcheck_blind.csv", BUILD / "spotcheck_blind.csv")
shutil.copy2(REPORTS / "spotcheck_key.csv", BUILD / "spotcheck_key.csv")

# 3. pool rows that could match a human row (same loose instruction)
rows = []
with open(ARM_A, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            key = loose(r["instruction"])
            if key in human_instr:
                rows.append({
                    "id": r["id"],
                    "loose_instruction": key,
                    "loose_response_hash": loose_hash(r["output"]),
                    "c": r.get("c"),
                    "q": r.get("q"),
                })
pool_matches = pd.DataFrame(rows)
pool_matches.to_csv(BUILD / "pool_matches.csv", index=False, encoding="utf-8-sig")
print(f"pool_matches.csv: {len(pool_matches)} candidate pool rows")

# how many human rows can reuse the production Stage 3 score?
exact = sum(
    1 for h in human
    if any(pm["loose_instruction"] == loose(h["instruction"])
           and pm["loose_response_hash"] == loose_hash(h["response"])
           for pm in pool_matches.to_dict("records"))
)
print(f"human rows with the SAME response in the pool (production score reusable): {exact}/100")
print(f"human rows needing a fresh judge pass: {100 - exact}/100")

# 4. dataset metadata
metadata = {
    "title": "bn-full-eval-validation",
    "id": "habiburrahman118/bn-full-eval-validation",
    "subtitle": "Judge-vs-human validation inputs (100 annotated rows) + blind spot-check sheet",
    "licenses": [{"name": "other"}],
}
(BUILD / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

total = sum(p.stat().st_size for p in BUILD.iterdir())
print(f"\ndataset built at {BUILD} ({total / 1e6:.3f} MB)")
for p in sorted(BUILD.iterdir()):
    print(f"  {p.name}: {p.stat().st_size / 1e6:.3f} MB")
