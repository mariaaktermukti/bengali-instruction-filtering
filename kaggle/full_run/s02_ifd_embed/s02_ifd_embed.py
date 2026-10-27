"""Full run, Stage 2 (GPU, Kaggle T4 x2): perplexity + IFD with the base model, and bge-m3 embeddings.

Same code as notebook cells 20 and 25 (Stages 6 and 8). The pilot took 207 s per 1K rows for IFD on one
T4 (~6 h for 100K), so the pool is split in two halves and each half is scored on its own GPU in a
separate process; the embeddings run afterwards on GPU 0.

Kaggle: GPU T4 x2, Internet ON, input = output of bn-full-s01-clean.
Outputs under /kaggle/working/bangla_filtering/full/scores/:
    ppl_ifd.parquet                  id, nll_cond, nll_uncond
    embeddings.npy + embedding_ids.json
    embedding_sanity.json
Scoring is checkpointed every chunk and stops after TIME_BUDGET_H, so a run that is cut short keeps
what it finished; attach this kernel's output to a rerun to continue (see find_previous below).
"""
import os, sys, glob, json, time, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working/bangla_filtering/full/scores")
OUT.mkdir(parents=True, exist_ok=True)
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "10.5"))
T0 = time.time()

BASE_MODEL, BASE_MODEL_FALLBACK = "meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"
EMBED_MODEL = "BAAI/bge-m3"
PROMPT_TMPL = "### নির্দেশনা:\n{instruction}\n\n### উত্তর:\n"     # identical to the pilot
SEED = 42


def find_input(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not under /kaggle/input: attach the bn-full-s01-clean output")
    return hits[0]


def find_previous(name):
    """Scores from an earlier (cut-short) run of this kernel, if its output is attached."""
    hits = sorted(glob.glob(f"/kaggle/input/**/full/scores/{name}", recursive=True))
    return pd.read_parquet(hits[0]) if hits else None


# ----------------------------------------------------------------------------------------------------
# worker: python s02_ifd_embed.py ifd <shard> <n_shards>   (CUDA_VISIBLE_DEVICES set by the parent)
# ----------------------------------------------------------------------------------------------------
def ifd_worker(shard, n_shards):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    DEVICE = "cuda"
    DTYPE = torch.float16                     # T4: no native bf16

    @torch.no_grad()
    def response_nll(model, tok, prompts, responses, max_prompt=512, max_resp=512, bs=8):
        seqs, starts = [], []
        for p, r in zip(prompts, responses):
            p_ids = tok(p, add_special_tokens=False)["input_ids"][-max_prompt:] if p else []
            r_ids = tok(r, add_special_tokens=False)["input_ids"][:max_resp]
            seqs.append([tok.bos_token_id] + p_ids + r_ids)
            starts.append(1 + len(p_ids))
        order = np.argsort([len(s) for s in seqs])
        out = np.full(len(seqs), np.nan)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(seqs[i])] = torch.tensor(seqs[i]); att[j, :len(seqs[i])] = 1
            logits = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE)).logits
            for j, i in enumerate(idx):
                st, en = starts[i], len(seqs[i])
                if en > st:
                    out[i] = torch.nn.functional.cross_entropy(logits[j, st - 1:en - 1].float(), ids[j, st:en].to(DEVICE)).item()
        return out

    for name in (BASE_MODEL, BASE_MODEL_FALLBACK):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            lm = AutoModelForCausalLM.from_pretrained(name, dtype=DTYPE, device_map=DEVICE).eval()
            break
        except Exception as e:
            print(f"cannot load {name}: {type(e).__name__}")
    print(f"[ifd {shard}] model {name}", flush=True)

    pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"), columns=["id", "instruction", "output"])
    pool = pool.iloc[shard::n_shards]
    path = OUT / f"ppl_ifd_shard{shard}.parquet"
    prev = find_previous("ppl_ifd.parquet")
    done = prev[prev["id"].isin(set(pool["id"]))] if prev is not None else pd.DataFrame({"id": []})
    todo = pool[~pool["id"].isin(set(done["id"]))]
    print(f"[ifd {shard}] {len(pool):,} rows, {len(done):,} already scored, {len(todo):,} to do", flush=True)

    chunk = 1_000
    for s in range(0, len(todo), chunk):
        if time.time() - T0 > TIME_BUDGET_H * 3600:
            print(f"[ifd {shard}] time budget reached, stopping at {len(done):,} rows", flush=True)
            break
        part = todo.iloc[s:s + chunk]
        prompts = [PROMPT_TMPL.format(instruction=i) for i in part["instruction"]]
        cond = response_nll(lm, tok, prompts, part["output"].tolist())
        uncond = response_nll(lm, tok, [""] * len(part), part["output"].tolist())
        done = pd.concat([done, pd.DataFrame({"id": part["id"].values, "nll_cond": cond, "nll_uncond": uncond})],
                         ignore_index=True)
        done.to_parquet(path, index=False)
        print(f"[ifd {shard}] {len(done):,} / {len(pool):,}  {(time.time() - T0) / 60:.0f} min", flush=True)
    done.to_parquet(path, index=False)


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
def embeddings(pool):
    import torch
    from sentence_transformers import SentenceTransformer

    emb_path, ids_path = OUT / "embeddings.npy", OUT / "embedding_ids.json"
    model = SentenceTransformer(EMBED_MODEL, device="cuda")
    model.max_seq_length = 512
    model.half()

    def embed(texts):
        return model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).astype(np.float32)

    t = time.time()
    E = embed((pool["instruction"] + "\n" + pool["output"].str[:1000]).tolist())
    np.save(emb_path, E); ids_path.write_text(json.dumps(pool["id"].tolist()))
    print(f"[embed] {len(pool):,} rows in {(time.time() - t) / 60:.1f} min", flush=True)

    # sanity check (notebook cell 25): MinHash near-dups vs random pairs
    rng = np.random.default_rng(SEED)
    nd = json.loads(Path(find_input("full/pool/near_dup_pairs.json")).read_text(encoding="utf-8"))
    pairs = [(a, b) for a, b in nd["pairs"] if a in nd["texts"] and b in nd["texts"]][:200]
    i, j = rng.integers(0, len(E), 2_000), rng.integers(0, len(E), 2_000)
    rand_sim = (E[i] * E[j]).sum(1)
    report = {"random_median": float(np.median(rand_sim)), "random_p95": float(np.quantile(rand_sim, .95)),
              "random_p99": float(np.quantile(rand_sim, .99)), "near_dup_pairs_checked": len(pairs)}
    if pairs:
        A = embed([nd["texts"][a] for a, _ in pairs]); B = embed([nd["texts"][b] for _, b in pairs])
        dup_sim = (A * B).sum(1)
        report.update(near_dup_median=float(np.median(dup_sim)),
                      near_dup_above_random_p99=float((dup_sim > report["random_p99"]).mean()))
    # nearest-neighbour similarity of a sample, to calibrate the DEITA threshold tau in Stage 4
    probe = rng.choice(len(E), min(5_000, len(E)), replace=False)
    Et = torch.from_numpy(E).cuda().half()
    nn = []
    for s in range(0, len(probe), 500):
        sims = Et[torch.from_numpy(probe[s:s + 500]).cuda()] @ Et.T
        sims[torch.arange(sims.shape[0]), torch.from_numpy(probe[s:s + 500]).cuda()] = -1
        nn.append(sims.max(1).values.float().cpu().numpy())
    nn = np.concatenate(nn)
    report["nn_sim_quantiles"] = {str(q): float(np.quantile(nn, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    (OUT / "embedding_sanity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ifd":
        ifd_worker(int(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers", "pyarrow"])
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("no HF_TOKEN secret, using the ungated base model mirror:", type(e).__name__)

    import torch
    n_gpu = torch.cuda.device_count()
    print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
    workers = [subprocess.Popen([sys.executable, __file__, "ifd", str(k), str(n_gpu)],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}) for k in range(n_gpu)]
    codes = [w.wait() for w in workers]
    print("ifd workers exited with", codes, flush=True)

    shards = [pd.read_parquet(p) for p in sorted(OUT.glob("ppl_ifd_shard*.parquet"))]
    ppl = pd.concat(shards, ignore_index=True).drop_duplicates("id")
    ppl.to_parquet(OUT / "ppl_ifd.parquet", index=False)
    for p in OUT.glob("ppl_ifd_shard*.parquet"):
        p.unlink()

    pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"), columns=["id", "instruction", "output", "domain"])
    m = pool.merge(ppl, on="id", how="left")
    m["ifd"] = np.exp(m["nll_cond"] - m["nll_uncond"])
    summary = {"pool_rows": len(pool), "ifd_scored": int(m["ifd"].notna().sum()),
               "ifd_ge_1": float((m["ifd"] >= 1).mean()),
               "ifd_median_by_domain": m.groupby("domain")["ifd"].median().round(3).to_dict(),
               "ifd_minutes": round((time.time() - T0) / 60, 1)}
    print(json.dumps(summary, indent=2), flush=True)

    embeddings(pool)
    summary["total_minutes"] = round((time.time() - T0) / 60, 1)
    (OUT / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    if summary["ifd_scored"] < len(pool):
        print(f"INCOMPLETE: {len(pool) - summary['ifd_scored']:,} rows without IFD; rerun with this output attached")
