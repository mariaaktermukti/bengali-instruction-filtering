"""Full run, Stage 4 (CPU only): judge report, the experimental arms A-F, arm diagnostics, blind spot-check.

Same logic as the pilot notebook (cell 22 tail, 23, 27-31), with two changes for the full pool:
  * the DEITA greedy step runs in numpy (fp32) instead of torch fp16 on a GPU;
  * a tau sweep first records, for each tau, how deep into the s-ranking the greedy step has to go to fill K
    rows, how many it rejects and how much arm C overlaps arm E. The arms are built with TAU_SIM (default 0.74;
    the notebook's 0.90 left C 91.5% identical to E on the full pool, see the v1/v2 notes in HANDOFF.md).

Kaggle: CPU kernel, inputs = outputs of bn-full-s01-clean, bn-full-s02-ifd-embed, bn-full-s02b-ifd-qwen (IFD for arm F)
and every bn-full-s03-judge-NN.
Outputs under /kaggle/working/bangla_filtering/full/:
    arms/arm_{A..F}.jsonl, arms/heldout.jsonl    training / eval files (format of notebook cell 29)
    pool/pool_scored.parquet                     pool + c, q, s, parse_ok, ifd, ppl
    reports/judge_report.json, tau_sweep.csv, arm_diagnostics.csv, spotcheck_blind.csv, spotcheck_key.csv,
    reports/stage4_manifest.json
"""
import os, re, glob, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INPUT = os.environ.get("INPUT_ROOT", "/kaggle/input")               # overridable for a local test
OUT = Path(os.environ.get("OUT_ROOT", "/kaggle/working/bangla_filtering/full"))
for sub in ["pool", "arms", "reports"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

SEED = 42
SUBSET_SIZE = int(os.environ.get("SUBSET_SIZE", "10000"))           # K, decided 2026-10-26
# v1 used the notebook's 0.90: C∩E 91.5%. 0.74 ≈ the pool's NN-sim p25 (0.746, Stage 2) and gives C∩E 58% (v2 sweep)
TAU_SIM = float(os.environ.get("TAU_SIM", "0.74"))
TAU_GRID = [0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.84, 0.88, 0.90, 0.92, 0.96]   # v1: C∩E 91.5% at 0.90
JUDGE_MIN_MASS = 0.5
MIN_JUDGE_COVERAGE = 0.99         # refuse to build arms on a judge run that is clearly unfinished
WORD = re.compile(r"[ঀ-৿‌‍A-Za-z0-9]+")                   # as in s01_clean.py
T0 = time.time()


def find_input(pattern):
    hits = sorted(glob.glob(f"{INPUT}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not under {INPUT}: attach the kernel that writes it")
    return hits[0]


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


# ====================================================================================================
# Load: pool + IFD + judge scores + embeddings, all aligned to pool order
# ====================================================================================================
pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"))
heldout = pd.read_parquet(find_input("full/pool/heldout.parquet"))

ppl = pd.read_parquet(find_input("full/scores/ppl_ifd.parquet"))          # Stage 2, Llama-3.2-1B
pool = pool.merge(ppl, on="id", how="left")
pool["ifd_llama"] = np.exp(pool["nll_cond"] - pool["nll_uncond"])
# Stage 2b: IFD with Qwen2.5-0.5B-Instruct, the model Stage 5 trains (Cherry-LLM scores with the tuned model).
# Arm F uses it when attached; otherwise it falls back to the Llama IFD.
qwen_files = sorted(glob.glob(f"{INPUT}/**/full/scores/ifd_qwen.parquet", recursive=True))
if qwen_files:
    qwen = pd.read_parquet(qwen_files[0]).rename(columns={"nll_cond": "nll_cond_qwen", "nll_uncond": "nll_uncond_qwen"})
    pool = pool.merge(qwen, on="id", how="left")
    pool["ifd_qwen"] = np.exp(pool["nll_cond_qwen"] - pool["nll_uncond_qwen"])
    pool["ppl"] = np.exp(pool["nll_uncond_qwen"])
    pool["ifd"] = pool["ifd_qwen"]
    IFD_SOURCE = "Qwen/Qwen2.5-0.5B-Instruct"
else:
    pool["ppl"] = np.exp(pool["nll_uncond"])
    pool["ifd"] = pool["ifd_llama"]
    IFD_SOURCE = "unsloth/Llama-3.2-1B"
print(f"IFD source: {IFD_SOURCE} | rows with IFD: {pool['ifd'].notna().sum():,}", flush=True)

# every judge session's output; the last one already holds all rows, but merging is harmless
judge_files = sorted(glob.glob(f"{INPUT}/**/full/scores/judge_en.parquet", recursive=True))
if not judge_files:
    raise FileNotFoundError("no judge_en.parquet: attach the bn-full-s03-judge-NN outputs")
scores = pd.concat([pd.read_parquet(p) for p in judge_files], ignore_index=True).drop_duplicates("id")
for p in judge_files:
    log(f"judge scores from {p}")
pool = pool.merge(scores, on="id", how="left")
coverage = float(pool["complexity_en"].notna().mean())
log(f"pool {len(pool):,} rows | IFD for {pool['ifd'].notna().sum():,} | judge for {pool['complexity_en'].notna().sum():,} ({coverage:.2%})")
if coverage < MIN_JUDGE_COVERAGE:
    raise RuntimeError(f"judge covers only {coverage:.2%} of the pool; finish Stage 3 first")

# notebook cell 22 (tail)
pool["parse_ok"] = (pool["complexity_en_mass"] > JUDGE_MIN_MASS) & (pool["quality_en_mass"] > JUDGE_MIN_MASS)
pool["c"] = pool["complexity_en"]; pool["q"] = pool["quality_en"]
pool["s"] = pool["c"] * pool["q"]                      # DEITA: complexity × quality (not q²)

emb_ids = json.loads(Path(find_input("full/scores/embedding_ids.json")).read_text())
E_raw = np.load(find_input("full/scores/embeddings.npy"))
pos = {i: k for k, i in enumerate(emb_ids)}
missing = [i for i in pool["id"] if i not in pos]
if missing:
    raise RuntimeError(f"{len(missing):,} pool rows have no embedding")
E = np.ascontiguousarray(E_raw[[pos[i] for i in pool["id"]]], dtype=np.float32)
del E_raw
log(f"embeddings {E.shape}, norms {np.linalg.norm(E[:1000], axis=1).min():.3f}-{np.linalg.norm(E[:1000], axis=1).max():.3f}")


# ====================================================================================================
# Judge report (notebook cell 23) + the q ceiling
# ====================================================================================================
judge_report = {
    "judge_files": judge_files,
    "coverage": round(coverage, 4),
    "parse_rate": round(float(pool["parse_ok"].mean()), 4),
    "c_mean": round(float(pool.c.mean()), 3), "q_mean": round(float(pool.q.mean()), 3),
    "spearman_c_vs_instr_len": round(spearmanr(pool.c, pool.instr_tokens, nan_policy="omit")[0], 3),
    "spearman_q_vs_resp_len":  round(spearmanr(pool.q, pool.resp_tokens, nan_policy="omit")[0], 3),
    "spearman_c_vs_q":         round(spearmanr(pool.c, pool.q, nan_policy="omit")[0], 3),
    "ifd_source": IFD_SOURCE,
    "spearman_q_vs_ifd":       round(spearmanr(pool.q, pool.ifd, nan_policy="omit")[0], 3),
    "spearman_ifd_vs_ifd_llama": round(spearmanr(pool.ifd, pool.ifd_llama, nan_policy="omit")[0], 3),
    "spearman_s_vs_c":         round(spearmanr(pool.s, pool.c, nan_policy="omit")[0], 3),
    "q_share_ge_4_9": round(float((pool.q >= 4.9).mean()), 4),
    "q_share_lt_4":   round(float((pool.q < 4).mean()), 4),
    "q_share_lt_3":   round(float((pool.q < 3).mean()), 4),
    "c_quantiles": {str(k): round(float(v), 3) for k, v in pool.c.quantile([.1, .25, .5, .75, .9]).items()},
    "q_quantiles": {str(k): round(float(v), 3) for k, v in pool.q.quantile([.1, .25, .5, .75, .9]).items()},
    "by_domain": pool.groupby("domain")[["c", "q", "s", "ifd"]].median().round(3).to_dict(orient="index"),
}
(OUT / "reports" / "judge_report.json").write_text(json.dumps(judge_report, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in judge_report.items() if k != "by_domain"}, indent=2), flush=True)
if judge_report["parse_rate"] < 0.95:
    raise RuntimeError("judge parse rate too low for a full run")
if abs(judge_report["spearman_c_vs_instr_len"]) > 0.7:
    print("⚠ complexity is mostly instruction length — report token counts per arm (Task 3 risk)")


# ====================================================================================================
# Arms (notebook cell 27), DEITA greedy in numpy
# ====================================================================================================
def deita_select(order, E, k, tau, chunk=2_048):
    """Exact greedy DEITA selection, vectorised per chunk. `order` = pool indices sorted by score (desc).
    Returns the selected indices and the depth, i.e. how many candidates of `order` were examined."""
    sel_idx, sel_emb, depth = [], E[:0], 0
    for st in range(0, len(order), chunk):
        cand = np.asarray(order[st:st + chunk])
        C = E[cand]
        prev_max = (C @ sel_emb.T).max(1) if len(sel_idx) else np.full(len(cand), -1.0)
        S = C @ C.T
        chosen = []
        for j in range(len(cand)):
            if prev_max[j] >= tau or (chosen and S[j, chosen].max() >= tau):
                continue
            chosen.append(j)
            if len(sel_idx) + len(chosen) >= k:
                break
        depth = st + (chosen[-1] + 1 if chosen and len(sel_idx) + len(chosen) >= k else len(cand))
        sel_idx += cand[chosen].tolist()
        sel_emb = np.concatenate([sel_emb, C[chosen]])
        if len(sel_idx) >= k:
            break
    return sel_idx, depth


def mean_pair_sim(ix, rng, n=2_000):
    S = E[rng.choice(ix, min(n, len(ix)), replace=False)]
    inner = S @ S.T
    return float((inner.sum() - np.trace(inner)) / (len(S) * (len(S) - 1)))


K = min(SUBSET_SIZE, len(pool))
rng = np.random.default_rng(SEED)
jitter = rng.uniform(0, 1e-6, len(pool))                    # deterministic tie-break
scored = np.where(pool["parse_ok"].values)[0]
log(f"K = {K:,} | scorable rows: {len(scored):,} / {len(pool):,}")
if len(scored) < K:
    raise RuntimeError(f"only {len(scored):,} rows have a usable judge score but arms need {K:,}")

ARMS = {"A": np.arange(len(pool)),
        "B": rng.choice(len(pool), K, replace=False)}
s_order = scored[np.argsort(-(pool["s"].values[scored] + jitter[scored]))]
E_set = set(s_order[:K].tolist())

# tau sweep: how strongly does the diversity step bite at each tau?
sweep_rng = np.random.default_rng(SEED + 1)
sweep, C_by_tau = [], {}
for tau in sorted(set(TAU_GRID + [TAU_SIM])):
    sel, depth = deita_select(s_order, E, K, tau)
    C_by_tau[tau] = (sel, depth)
    sub = pool.iloc[sel]
    sweep.append({"tau": tau, "n_selected": len(sel), "depth": depth, "rejected": depth - len(sel),
                  "rejected_frac": round((depth - len(sel)) / max(1, depth), 4),
                  "overlap_with_E": round(len(E_set & set(sel)) / K, 4),
                  "s_min": round(float(sub.s.min()), 3), "c_mean": round(float(sub.c.mean()), 3),
                  "q_mean": round(float(sub.q.mean()), 3), "mean_pair_sim": round(mean_pair_sim(np.asarray(sel), sweep_rng), 4)})
    log(f"tau {tau:.2f}: {sweep[-1]}")
sweep = pd.DataFrame(sweep)
sweep.to_csv(OUT / "reports" / "tau_sweep.csv", index=False)
print(sweep.to_string(index=False), flush=True)

tau = TAU_SIM
ARMS["C"], c_depth = C_by_tau[tau]
while len(ARMS["C"]) < K and tau < 0.99:                    # τ too strict for this pool → relax and log it
    tau = round(tau + 0.02, 2)
    print(f"C reached only {len(ARMS['C']):,} rows; relaxing τ to {tau}")
    ARMS["C"], c_depth = deita_select(s_order, E, K, tau)
TAU_USED = tau

ARMS["D"] = scored[np.argsort(-(pool["q"].values[scored] + jitter[scored]))][:K]
ARMS["E"] = s_order[:K]
ok = np.where(pool["parse_ok"].values & (pool["ifd"].values < 1))[0]
ARMS["F"] = ok[np.argsort(-(pool["ifd"].values[ok] + jitter[ok]))][:K]

ARMS = {a: np.asarray(ix, dtype=np.int64) for a, ix in ARMS.items()}
q_cut = float(pool["q"].values[ARMS["D"]].min())
overlaps = {f"{a}∩{b}": round(len(set(ARMS[a]) & set(ARMS[b])) / K, 4)
            for a, b in [("C", "E"), ("C", "D"), ("C", "B"), ("D", "E"), ("C", "F"), ("E", "F")]}
log(f"{ {a: len(ix) for a, ix in ARMS.items()} } | τ used: {TAU_USED} | C depth {c_depth:,}")
print("overlaps:", overlaps)
print(f"arm D: K-th q = {q_cut:.3f}; rows with q ≥ that: {(pool.q >= q_cut).sum():,} "
      f"(> K means D's cut falls inside the q ceiling and is partly decided by tiny q differences)", flush=True)


# ====================================================================================================
# Arm diagnostics (notebook cell 28)
# ====================================================================================================
def distinct_n(texts, n=2, sample=3_000):
    texts = list(texts)[:sample]
    grams, total = set(), 0
    for t in texts:
        w = WORD.findall(t)
        g = list(zip(*[w[i:] for i in range(n)]))
        grams.update(g); total += len(g)
    return len(grams) / total if total else 0.0

probe = rng.choice(len(pool), min(3_000, len(pool)), replace=False)
E_probe = E[probe]

def arm_diag(name, ix):
    sub = pool.iloc[ix]
    cov = np.full(len(probe), -1.0, dtype=np.float32)
    for st in range(0, len(ix), 20_000):                    # arm A is the whole pool: block the product
        cov = np.maximum(cov, (E_probe @ E[ix[st:st + 20_000]].T).max(1))
    shuffled = sub.sample(frac=1, random_state=SEED)
    d = {"arm": name, "n": len(ix),
         "tokens_total": int(sub.instr_tokens.sum() + sub.resp_tokens.sum()),
         "resp_tokens_mean": round(sub.resp_tokens.mean(), 1), "instr_tokens_mean": round(sub.instr_tokens.mean(), 1),
         "c_mean": round(sub.c.mean(), 3), "q_mean": round(sub.q.mean(), 3),
         "distinct2_resp": round(distinct_n(shuffled.output), 4),
         "coverage": round(float(cov.mean()), 4), "mean_pair_sim": round(mean_pair_sim(ix, rng), 4),
         "ifd_median": round(sub.ifd.median(), 3)}
    d.update({f"dom_{k}": round(v, 3) for k, v in sub.domain.value_counts(normalize=True).items()})
    return d

diag = pd.DataFrame([arm_diag(a, ix) for a, ix in ARMS.items()]).fillna(0)
diag.to_csv(OUT / "reports" / "arm_diagnostics.csv", index=False)
print(diag.set_index("arm").T.to_string(), flush=True)
print("coverage = mean cos-sim from random pool rows to their nearest arm row (higher = arm spans the pool)")
print("mean_pair_sim = average similarity inside the arm (lower = more diverse)")


# ====================================================================================================
# Write arms (notebook cell 29)
# ====================================================================================================
def to_record(r, arm):
    return {"id": r["id"], "arm": arm,
            "messages": [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["output"]}],
            "instruction": r["instruction"], "output": r["output"], "domain": r["domain"],
            "c": None if pd.isna(r.get("c")) else round(float(r["c"]), 4),
            "q": None if pd.isna(r.get("q")) else round(float(r["q"]), 4)}

for arm, ix in ARMS.items():
    sub = pool.iloc[ix]
    with open(OUT / "arms" / f"arm_{arm}.jsonl", "w", encoding="utf-8") as f:
        for _, r in sub.iterrows():
            f.write(json.dumps(to_record(r, arm), ensure_ascii=False) + "\n")
with open(OUT / "arms" / "heldout.jsonl", "w", encoding="utf-8") as f:
    for _, r in heldout.iterrows():
        f.write(json.dumps(to_record(r, "heldout"), ensure_ascii=False) + "\n")
pool.to_parquet(OUT / "pool" / "pool_scored.parquet", index=False)
log(f"arms written -> {OUT / 'arms'}")


# ====================================================================================================
# Blind spot-check (notebook cell 31) + manifest
# ====================================================================================================
n_sc = min(50, K)
sc = pd.concat([pool.iloc[rng.choice(ARMS["C"], n_sc, replace=False)].assign(arm="C"),
                pool.iloc[rng.choice(ARMS["B"], n_sc, replace=False)].assign(arm="B")])
sc = sc.sample(frac=1, random_state=SEED).reset_index(drop=True)
sc["item"] = np.arange(len(sc))
sc[["item", "instruction", "output"]].assign(rating_1to5="", notes="").to_csv(
    OUT / "reports" / "spotcheck_blind.csv", index=False, encoding="utf-8-sig")
sc[["item", "id", "arm", "c", "q"]].to_csv(OUT / "reports" / "spotcheck_key.csv", index=False)

for _, r in pool[pool.parse_ok].nsmallest(3, "s").iterrows():
    print(f"\n[s={r.s:.2f} c={r.c:.2f} q={r.q:.2f}] {r.instruction[:150]}\n→ {r.output[:200]}")

manifest = {
    "subset_size": K, "ifd_source": IFD_SOURCE, "tau_requested": TAU_SIM, "tau_used": TAU_USED, "c_depth": c_depth,
    "pool_rows_clean": len(pool), "heldout_rows": len(heldout), "scorable_rows": len(scored),
    "arms": {a: len(ix) for a, ix in ARMS.items()}, "overlaps": overlaps,
    "arm_D_q_cut": round(q_cut, 4), "rows_q_ge_D_cut": int((pool.q >= q_cut).sum()),
    "tau_sweep": sweep.to_dict(orient="records"),
    "judge_report": {k: v for k, v in judge_report.items() if k != "by_domain"},
    "total_minutes": round((time.time() - T0) / 60, 1),
}
(OUT / "reports" / "stage4_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                                                      encoding="utf-8")
print(json.dumps({k: v for k, v in manifest.items() if k not in ("tau_sweep", "judge_report")}, indent=2,
                 ensure_ascii=False, default=str))
