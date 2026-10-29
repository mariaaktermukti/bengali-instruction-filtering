"""Full run, Stage 2b (GPU, Kaggle T4 x2): IFD recomputed with the model we fine-tune, Qwen2.5-0.5B-Instruct.

Stage 2 computed IFD with Llama-3.2-1B, but Stage 5 trains Qwen2.5-0.5B-Instruct (as the pilot did), and Cherry-LLM
computes IFD with the model being tuned. Same code as Stage 2 (notebook cell 20), with three changes:
  * the conditional prompt is Qwen's chat template, exactly as in scripts/kaggle_lora_training.py;
  * Qwen has no BOS token, so <|endoftext|> takes its place as the first token (cond and uncond alike);
  * fp32 weights + fp16 autocast, the training precision (Qwen2.5 in pure fp16 can overflow).
Embeddings are not recomputed.

Kaggle: GPU T4 x2, Internet ON, input = output of bn-full-s01-clean (+ an earlier bn-full-s02b-ifd-qwen to resume).
Outputs under /kaggle/working/bangla_filtering/full/scores/:
    ifd_qwen.parquet        id, nll_cond, nll_uncond
    ifd_qwen_summary.json   rows scored, NaN count, IFD >= 1 share, median by domain, minutes
TIME_BUDGET_H is 1.75 h by default because the account had 2.5 h of GPU quota left (2026-10-30); the output is
checkpointed every chunk, so a cut-short run keeps what it finished.
"""
import os, sys, glob, json, time, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working/bangla_filtering/full/scores")
OUT.mkdir(parents=True, exist_ok=True)
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "1.75"))
T0 = time.time()

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"         # = MODEL_NAME in scripts/kaggle_lora_training.py
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
# worker: python s02b_ifd_qwen.py ifd <shard> <n_shards>   (CUDA_VISIBLE_DEVICES set by the parent)
# ----------------------------------------------------------------------------------------------------
def ifd_worker(shard, n_shards):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    DEVICE = "cuda"

    @torch.no_grad()
    def response_nll(model, tok, prompts, responses, max_prompt=512, max_resp=512, bs=8):
        start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.convert_tokens_to_ids("<|endoftext|>")
        seqs, starts = [], []
        for p, r in zip(prompts, responses):
            p_ids = tok(p, add_special_tokens=False)["input_ids"][-max_prompt:] if p else []
            r_ids = tok(r, add_special_tokens=False)["input_ids"][:max_resp]
            seqs.append([start_id] + p_ids + r_ids)
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
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids.to(DEVICE), attention_mask=att.to(DEVICE)).logits
            for j, i in enumerate(idx):
                st, en = starts[i], len(seqs[i])
                if en > st:
                    out[i] = torch.nn.functional.cross_entropy(logits[j, st - 1:en - 1].float(), ids[j, st:en].to(DEVICE)).item()
        return out

    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map=DEVICE).eval()
    print(f"[ifd {shard}] model {MODEL}", flush=True)

    pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"), columns=["id", "instruction", "output"])
    pool = pool.iloc[shard::n_shards]
    path = OUT / f"ifd_qwen_shard{shard}.parquet"
    prev = find_previous("ifd_qwen.parquet")
    done = prev[prev["id"].isin(set(pool["id"]))] if prev is not None else pd.DataFrame({"id": []})
    todo = pool[~pool["id"].isin(set(done["id"]))]
    print(f"[ifd {shard}] {len(pool):,} rows, {len(done):,} already scored, {len(todo):,} to do", flush=True)

    chunk = 1_000
    for s in range(0, len(todo), chunk):
        if time.time() - T0 > TIME_BUDGET_H * 3600:
            print(f"[ifd {shard}] time budget reached, stopping at {len(done):,} rows", flush=True)
            break
        part = todo.iloc[s:s + chunk]
        prompts = [tok.apply_chat_template([{"role": "user", "content": i}], tokenize=False, add_generation_prompt=True)
                   for i in part["instruction"]]
        cond = response_nll(lm, tok, prompts, part["output"].tolist())
        uncond = response_nll(lm, tok, [""] * len(part), part["output"].tolist())
        done = pd.concat([done, pd.DataFrame({"id": part["id"].values, "nll_cond": cond, "nll_uncond": uncond})],
                         ignore_index=True)
        done.to_parquet(path, index=False)
        print(f"[ifd {shard}] {len(done):,} / {len(pool):,}  {(time.time() - T0) / 60:.0f} min", flush=True)
    done.to_parquet(path, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ifd":
        ifd_worker(int(sys.argv[2]), int(sys.argv[3]))
        sys.exit(0)

    import torch
    n_gpu = torch.cuda.device_count()
    print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)
    workers = [subprocess.Popen([sys.executable, __file__, "ifd", str(k), str(n_gpu)],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}) for k in range(n_gpu)]
    codes = [w.wait() for w in workers]
    print("ifd workers exited with", codes, flush=True)

    shards = [pd.read_parquet(p) for p in sorted(OUT.glob("ifd_qwen_shard*.parquet"))]
    ifd = pd.concat(shards, ignore_index=True).drop_duplicates("id")
    ifd.to_parquet(OUT / "ifd_qwen.parquet", index=False)
    for p in OUT.glob("ifd_qwen_shard*.parquet"):
        p.unlink()

    pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"), columns=["id", "domain"])
    m = pool.merge(ifd, on="id", how="left")
    m["ifd"] = np.exp(m["nll_cond"] - m["nll_uncond"])
    summary = {"model": MODEL, "pool_rows": len(pool), "ifd_scored": int(m["ifd"].notna().sum()),
               "nan_or_missing": int(m["ifd"].isna().sum()),
               "ifd_ge_1": float((m["ifd"] >= 1).mean()),
               "ifd_median": float(m["ifd"].median()),
               "ifd_median_by_domain": m.groupby("domain")["ifd"].median().round(3).to_dict(),
               "total_minutes": round((time.time() - T0) / 60, 1)}
    llama = find_previous("ppl_ifd.parquet")           # Stage 2 output, if attached: how different is the ranking?
    if llama is not None:
        mm = m.merge(llama, on="id", suffixes=("", "_llama"))
        ifd_llama = np.exp(mm["nll_cond_llama"] - mm["nll_uncond_llama"])
        summary["spearman_vs_llama_ifd"] = round(float(mm["ifd"].corr(ifd_llama, method="spearman")), 4)
    (OUT / "ifd_qwen_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if summary["ifd_scored"] < len(pool):
        print(f"INCOMPLETE: {len(pool) - summary['ifd_scored']:,} rows without IFD; rerun with this output attached")
