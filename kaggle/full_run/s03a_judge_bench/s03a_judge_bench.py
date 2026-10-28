"""Full run, Stage 3a (GPU, Kaggle T4 x2): judge throughput benchmark before the full Stage 3 judge.

Scores the pilot rows with the English rubric (complexity c, quality q; notebook cell 22) on several
backends and compares speed and agreement with the pilot scores (HF backend, fp16, both T4s):
    hf          the pilot code: Qwen2.5-7B-Instruct fp16, device_map="auto" over both GPUs   (N_HF rows)
    vllm_tp2    vLLM, same fp16 weights, tensor-parallel over both GPUs                      (all rows)
    awq_dp2     vLLM, Qwen2.5-7B-Instruct-AWQ (4-bit), one copy per GPU, half the rows each  (all rows)
    gptq8_dp2   vLLM, Qwen2.5-7B-Instruct-GPTQ-Int8, one copy per GPU                        (all rows)
Each backend runs in its own subprocess (vLLM is pip-installed after the HF run, and a script kernel
cannot restart), so one backend failing is recorded in the report and the others still run.

Kaggle: GPU T4 x2, Internet ON, input = output of bangla-filtering-selection (the pilot notebook).
Outputs under /kaggle/working/bangla_filtering/full/judge_bench/:
    judge_bench.json            per backend: rows/h, est. hours for the 91,120-row pool, Spearman vs pilot
    judge_bench_scores.parquet  id + c, q per backend + pilot c, q
Decision rule (PLAN.md): the fastest backend whose Spearman with the pilot c and q are both >= 0.9.
"""
import os, sys, glob, json, math, time, subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working/bangla_filtering/full/judge_bench")
OUT.mkdir(parents=True, exist_ok=True)
ROWS_PATH = OUT / "bench_rows.parquet"
SEED = 42
N_HF = int(os.environ.get("N_HF", "300"))          # the HF backend is slow (~1.7 s per row): a subset
FULL_POOL_ROWS = 91_120                             # Stage 1 clean pool
MIN_SPEARMAN = 0.9
JUDGE_MAX_INSTR_CHARS, JUDGE_MAX_RESP_CHARS = 1_200, 2_000
MAX_MODEL_LEN = 8_192

BACKENDS = {   # name: (model, tensor_parallel, data_parallel shards)
    "vllm_tp2":  ("Qwen/Qwen2.5-7B-Instruct", 2, 1),
    "awq_dp2":   ("Qwen/Qwen2.5-7B-Instruct-AWQ", 1, 2),
    "gptq8_dp2": ("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", 1, 2),
}
HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# identical to the pilot notebook (cell 22), English rubric only
RUBRIC = {
 "complexity": """You are rating Bengali (বাংলা) instructions for an instruction-tuning dataset.
Rate how COMPLEX / DEMANDING the instruction is, from 1 to 6:
1 = trivial (one fact, one-word answer, copying)
2 = simple (one step, common knowledge)
3 = moderate (a few steps or a short explanation)
4 = involved (multi-step reasoning, several constraints, or specialised knowledge)
5 = hard (deep reasoning, interacting constraints, long structured output)
6 = very hard (expert-level, careful multi-stage reasoning)
Judge only the difficulty of the task, not the writing quality. Do not translate it.

Instruction:
<<<
{instruction}
>>>

Reply with one digit from 1 to 6.""",
 "quality": """You are rating a Bengali (বাংলা) instruction–response pair for an instruction-tuning dataset.
Rate the QUALITY of the RESPONSE from 1 to 6, considering:
- correctness and factual accuracy
- whether it follows every part of the instruction (format, length, all sub-questions)
- helpfulness and appropriate depth
- Bengali language quality: natural, fluent, grammatical; penalise machine-translation artefacts
  (literal English word order, untranslated English, broken conjuncts, wrong script)
1 = wrong, off-topic or unusable   2 = major errors   3 = partly correct or poorly written
4 = correct and acceptable   5 = correct, complete and fluent   6 = excellent, expert native-speaker quality

Instruction:
<<<
{instruction}
>>>

Response:
<<<
{response}
>>>

Reply with one digit from 1 to 6.""",
}
DIGITS = [str(d) for d in range(1, 7)]


def find_input(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not under /kaggle/input: attach the bangla-filtering-selection output")
    return hits[0]


def clip(s, n):
    return s if len(s) <= n else s[:n]


def clip_middle(s, n):        # keep the start and the final answer (math answers sit at the end)
    return s if len(s) <= n else s[: int(n * 0.7)] + "\n…\n" + s[-int(n * 0.3):]


def judge_messages(part, kind):
    return [RUBRIC[kind].format(instruction=clip(i, JUDGE_MAX_INSTR_CHARS), response=clip_middle(o, JUDGE_MAX_RESP_CHARS))
            for i, o in zip(part["instruction"], part["output"])]


def chat_prompts(tok, messages):
    return [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True)
            for m in messages]


def expectation(dists):
    mass = np.array([sum(p.values()) for p in dists])
    expected = np.array([sum(int(d) * v for d, v in p.items()) / m if m > 0 else np.nan for p, m in zip(dists, mass)])
    return expected, mass


def save_worker(name, rows, res, load_s, score_s, max_prompt_tokens):
    pd.DataFrame(res).to_parquet(OUT / f"scores_{name}.parquet", index=False)
    (OUT / f"timing_{name}.json").write_text(json.dumps(
        {"rows": len(rows), "load_s": round(load_s, 1), "score_s": round(score_s, 1),
         "max_prompt_tokens": int(max_prompt_tokens)}))
    print(f"[{name}] {len(rows)} rows: load {load_s:.0f}s, score {score_s:.0f}s", flush=True)


# ----------------------------------------------------------------------------------------------------
# worker: python s03a_judge_bench.py hf
# ----------------------------------------------------------------------------------------------------
def hf_worker():
    """DigitJudge, HF backend, exactly as in the pilot (fp16 on T4, device_map='auto' over both GPUs)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = pd.read_parquet(ROWS_PATH).head(N_HF)
    t = time.time()
    tok = AutoTokenizer.from_pretrained(HF_MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.float16, device_map="auto").eval()
    digit_ids = {d: sorted({tok.encode(d, add_special_tokens=False)[-1],
                            tok.encode(" " + d, add_special_tokens=False)[-1]}) for d in DIGITS}
    load_s = time.time() - t

    def score(prompts, bs=16, max_tokens=6_144):
        lens = [len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]]
        order = np.argsort(lens)[::-1]
        dists = [None] * len(prompts)
        b = 0
        while b < len(order):
            idx = order[b:b + max(1, min(bs, max_tokens // lens[order[b]]))]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                probs = model(**enc, logits_to_keep=1).logits[:, -1].float().softmax(-1).cpu()
            for i, row in zip(idx, probs):
                dists[i] = {d: float(row[ids].sum()) for d, ids in digit_ids.items()}
            b += len(idx)
        return expectation(dists), max(lens)

    t = time.time()
    res, max_len = {"id": rows["id"].values}, 0
    for kind, col in (("complexity", "c"), ("quality", "q")):
        (res[col], res[f"{col}_mass"]), m = score(chat_prompts(tok, judge_messages(rows, kind)))
        max_len = max(max_len, m)
    save_worker("hf", rows, res, load_s, time.time() - t, max_len)


# ----------------------------------------------------------------------------------------------------
# worker: python s03a_judge_bench.py vllm <backend> <shard> <n_shards>
# ----------------------------------------------------------------------------------------------------
def vllm_worker(backend, shard, n_shards):
    from vllm import LLM, SamplingParams

    model_name, tp, _ = BACKENDS[backend]
    rows = pd.read_parquet(ROWS_PATH).iloc[shard::n_shards]
    t = time.time()
    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=tp, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, seed=SEED, enforce_eager=os.environ.get("VLLM_EAGER") == "1")
    tok = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    load_s = time.time() - t

    t = time.time()
    res, max_len = {"id": rows["id"].values}, 0
    for kind, col in (("complexity", "c"), ("quality", "q")):
        prompts = chat_prompts(tok, judge_messages(rows, kind))
        max_len = max(max_len, max(len(x) for x in tok(prompts, add_special_tokens=False)["input_ids"]))
        dists = []
        for o in llm.generate(prompts, params, use_tqdm=False):
            p = defaultdict(float)
            for tid, lp in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).items():
                tok_str = lp.decoded_token if getattr(lp, "decoded_token", None) is not None else tok.decode([tid])
                if tok_str.strip() in DIGITS:
                    p[tok_str.strip()] += math.exp(lp.logprob)
            dists.append(p)
        res[col], res[f"{col}_mass"] = expectation(dists)
    save_worker(f"{backend}_s{shard}", rows, res, load_s, time.time() - t, max_len)


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
def run(args, timeout_min, env=None, log=None):
    """Run a worker; return (ok, last lines of its output)."""
    log = log or OUT / f"log_{'_'.join(args)}.txt"
    with open(log, "w") as f:
        try:
            code = subprocess.run([sys.executable, __file__, *args], env={**os.environ, **(env or {})},
                                  stdout=f, stderr=subprocess.STDOUT, timeout=timeout_min * 60).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    tail = Path(log).read_text(errors="replace").splitlines()[-25:]
    print(f"--- {' '.join(args)}: exit {code}\n" + "\n".join(tail), flush=True)
    return code == 0, "\n".join(tail)


def run_vllm(backend):
    """All shards of a backend at once (one per GPU for dp2); retried once in eager mode on failure."""
    _, tp, n_shards = BACKENDS[backend]
    for attempt, extra in enumerate(({}, {"VLLM_EAGER": "1"})):
        t = time.time()
        procs = []
        for k in range(n_shards):
            env = {**os.environ, **extra, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
            if n_shards > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(k)
            log = open(OUT / f"log_{backend}_s{k}_try{attempt}.txt", "w")
            procs.append((subprocess.Popen([sys.executable, __file__, "vllm", backend, str(k), str(n_shards)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT), log))
        codes = []
        for p, log in procs:
            try:
                codes.append(p.wait(timeout=45 * 60))
            except subprocess.TimeoutExpired:
                p.kill(); codes.append("timeout")
            log.close()
        wall = time.time() - t
        tails = [Path(log.name).read_text(errors="replace").splitlines()[-25:] for _, log in procs]
        print(f"--- {backend} try {attempt}: exit {codes}, {wall:.0f}s\n" + "\n".join(tails[0]), flush=True)
        if all(c == 0 for c in codes):
            return True, "", wall, bool(extra)
    return False, "\n".join(tails[0]), wall, True


def summarise(name, scores, timings, pilot, hf):
    """Speed and agreement for one backend."""
    from scipy.stats import spearmanr
    m = scores.merge(pilot, on="id", suffixes=("", "_pilot"))
    rows = sum(t["rows"] for t in timings)
    score_s = max(t["score_s"] for t in timings)        # shards run in parallel: wall time is the slowest
    rep = {"rows": rows, "load_s": max(t["load_s"] for t in timings), "score_s": score_s,
           "rows_per_h": round(rows / score_s * 3600), "est_hours_full_pool": round(FULL_POOL_ROWS / (rows / score_s) / 3600, 1),
           "max_prompt_tokens": max(t["max_prompt_tokens"] for t in timings),
           "parse_rate": round(float(((scores.c_mass > 0.5) & (scores.q_mass > 0.5)).mean()), 4)}
    for col in ("c", "q"):
        rep[f"spearman_{col}_vs_pilot"] = round(float(spearmanr(m[col], m[f"{col}_pilot"], nan_policy="omit")[0]), 4)
        rep[f"mae_{col}_vs_pilot"] = round(float((m[col] - m[f"{col}_pilot"]).abs().mean()), 4)
        if hf is not None and name != "hf":
            h = scores.merge(hf, on="id", suffixes=("", "_hf"))
            rep[f"spearman_{col}_vs_hf_rerun"] = round(float(spearmanr(h[col], h[f"{col}_hf"], nan_policy="omit")[0]), 4)
    return rep


def main():
    T0 = time.time()
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("no HF_TOKEN secret (Qwen is ungated, only rate limits differ):", type(e).__name__)

    pilot = pd.read_parquet(find_input("pilot/pool/pool_scored.parquet"),
                            columns=["id", "instruction", "output", "domain", "complexity_en", "quality_en"])
    pilot = pilot.sample(frac=1, random_state=SEED).reset_index(drop=True)   # head(N_HF) = random subset
    pilot.to_parquet(ROWS_PATH, index=False)
    pilot_cq = pilot[["id"]].assign(c=pilot.complexity_en, q=pilot.quality_en)
    print(f"{len(pilot)} pilot rows, HF backend on the first {N_HF}", flush=True)

    report, all_scores = {"pilot_rows": len(pilot), "n_hf": N_HF, "full_pool_rows": FULL_POOL_ROWS}, {"pilot": pilot_cq}
    hf = None

    ok, tail = run(["hf"], timeout_min=60)
    if ok:
        hf = pd.read_parquet(OUT / "scores_hf.parquet")
        report["hf"] = summarise("hf", hf, [json.loads((OUT / "timing_hf.json").read_text())], pilot_cq, None)
        all_scores["hf"] = hf
    else:
        report["hf"] = {"status": "failed", "error_tail": tail}

    t = time.time()
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
    report["vllm_install"] = {"ok": pip.returncode == 0, "seconds": round(time.time() - t),
                              "stderr_tail": pip.stderr[-2_000:]}
    # vllm upgrades torch (v1: 2.13+cu130) but leaves Kaggle's cu128 torchaudio, which transformers then imports and
    # crashes on. vllm does not need torchaudio, so remove it.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], capture_output=True, text=True)
    ver = subprocess.run([sys.executable, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__, "
                          "torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"],
                         capture_output=True, text=True)
    report["vllm_install"]["versions"] = (ver.stdout or ver.stderr).strip()[-500:]
    print("vllm install:", report["vllm_install"]["ok"], report["vllm_install"]["versions"], flush=True)

    for backend in BACKENDS:
        if not report["vllm_install"]["ok"]:
            report[backend] = {"status": "skipped: vllm install failed"}
            continue
        ok, tail, wall, eager = run_vllm(backend)
        if not ok:
            report[backend] = {"status": "failed", "error_tail": tail}
            continue
        n = BACKENDS[backend][2]
        sc = pd.concat([pd.read_parquet(OUT / f"scores_{backend}_s{k}.parquet") for k in range(n)], ignore_index=True)
        tm = [json.loads((OUT / f"timing_{backend}_s{k}.json").read_text()) for k in range(n)]
        report[backend] = {**summarise(backend, sc, tm, pilot_cq, hf), "wall_s_incl_load": round(wall), "eager": eager}
        all_scores[backend] = sc
        (OUT / "judge_bench.json").write_text(json.dumps(report, indent=2))    # keep partial results

    merged = pilot_cq.rename(columns={"c": "c_pilot", "q": "q_pilot"})
    for name, sc in all_scores.items():
        if name != "pilot":
            merged = merged.merge(sc[["id", "c", "q"]].rename(columns={"c": f"c_{name}", "q": f"q_{name}"}), on="id", how="left")
    merged.to_parquet(OUT / "judge_bench_scores.parquet", index=False)

    ok_backends = {k: v for k, v in report.items() if isinstance(v, dict) and "rows_per_h" in v
                   and v["spearman_c_vs_pilot"] >= MIN_SPEARMAN and v["spearman_q_vs_pilot"] >= MIN_SPEARMAN
                   and v["parse_rate"] >= 0.95}
    report["recommended"] = max(ok_backends, key=lambda k: ok_backends[k]["rows_per_h"]) if ok_backends else None
    report["total_minutes"] = round((time.time() - T0) / 60, 1)
    (OUT / "judge_bench.json").write_text(json.dumps(report, indent=2))
    for p in OUT.glob("timing_*.json"):
        p.unlink()

    print(json.dumps({k: v for k, v in report.items() if not (isinstance(v, dict) and "error_tail" in v)}, indent=2))
    print(f"\n{'backend':<11}{'rows/h':>9}{'h/91K':>8}{'rho c':>8}{'rho q':>8}{'parse':>8}")
    for k in ["hf", *BACKENDS]:
        v = report.get(k, {})
        if "rows_per_h" in v:
            print(f"{k:<11}{v['rows_per_h']:>9,}{v['est_hours_full_pool']:>8}{v['spearman_c_vs_pilot']:>8}"
                  f"{v['spearman_q_vs_pilot']:>8}{v['parse_rate']:>8}")
        else:
            print(f"{k:<11}  {v.get('status', '?')}")
    print("RECOMMENDED:", report["recommended"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hf":
        hf_worker()
    elif len(sys.argv) > 1 and sys.argv[1] == "vllm":
        vllm_worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        main()
