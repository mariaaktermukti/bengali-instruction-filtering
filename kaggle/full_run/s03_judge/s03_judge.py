"""Full run, Stage 3 (GPU, Kaggle T4 x2): LLM judge, complexity c and quality q for every pool row.

Same rubric and scoring as the pilot (notebook cell 22, English rubric only): Qwen2.5-7B-Instruct fp16,
expected digit 1-6 from the next-token distribution. Backend = vLLM, tensor-parallel over both T4s
(`vllm_tp2` in Stage 3a: 4,374 rows/h, Spearman 1.000 vs the pilot HF scores; ~20.8 h for 91,120 rows).

The pool does not fit in one 12 h session, so each session is its own kernel (bn-full-s03-judge-01, -02, ...),
and each attaches bn-full-s01-clean plus the outputs of all earlier judge kernels (see find_previous).
Rows are scored in a fixed shuffled order, so a partial result is a random sample of the pool.

Kaggle: GPU T4 x2, Internet ON, input = output of bn-full-s01-clean (+ earlier bn-full-s03-judge-NN).
Outputs under /kaggle/working/bangla_filtering/full/scores/:
    judge_en.parquet     id, complexity_en, complexity_en_mass, quality_en, quality_en_mass (all rows so far)
    judge_summary.json   rows done / remaining, rows/h this session, parse rate, medians by domain
Scoring is checkpointed every chunk and stops after TIME_BUDGET_H.
"""
import os, sys, glob, json, math, time, subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/kaggle/working/bangla_filtering/full/scores")
OUT.mkdir(parents=True, exist_ok=True)
SCORES = OUT / "judge_en.parquet"
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "10.5"))
SEED = 42
CHUNK = 2_000                                       # ~27 min at the Stage 3a rate
JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VLLM_VERSION = "0.32.0"                             # the version Stage 3a validated
JUDGE_MAX_INSTR_CHARS, JUDGE_MAX_RESP_CHARS = 1_200, 2_000
JUDGE_MIN_MASS = 0.5
MAX_MODEL_LEN = 8_192
COLS = ["id", "complexity_en", "complexity_en_mass", "quality_en", "quality_en_mass"]

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
        raise FileNotFoundError(f"{pattern} not under /kaggle/input: attach the bn-full-s01-clean output")
    return hits[0]


def find_previous():
    """Scores from earlier judge kernels (attached inputs) and from earlier attempts in this session."""
    hits = sorted(glob.glob("/kaggle/input/**/full/scores/judge_en.parquet", recursive=True))
    if SCORES.exists():
        hits.append(str(SCORES))
    parts = [pd.read_parquet(p, columns=COLS) for p in hits]
    for p, df in zip(hits, parts):
        print(f"previous scores: {len(df):,} rows from {p}", flush=True)
    return pd.concat(parts, ignore_index=True).drop_duplicates("id") if parts else pd.DataFrame(columns=COLS)


def clip(s, n):
    return s if len(s) <= n else s[:n]


def clip_middle(s, n):        # keep the start and the final answer (math answers sit at the end)
    return s if len(s) <= n else s[: int(n * 0.7)] + "\n…\n" + s[-int(n * 0.3):]


def judge_messages(part, kind):
    return [RUBRIC[kind].format(instruction=clip(i, JUDGE_MAX_INSTR_CHARS), response=clip_middle(o, JUDGE_MAX_RESP_CHARS))
            for i, o in zip(part["instruction"], part["output"])]


def expectation(dists):
    mass = np.array([sum(p.values()) for p in dists])
    expected = np.array([sum(int(d) * v for d, v in p.items()) / m if m > 0 else np.nan for p, m in zip(dists, mass)])
    return expected, mass


# ----------------------------------------------------------------------------------------------------
# worker: python s03_judge.py worker   (DEADLINE = epoch seconds, set by the parent)
# ----------------------------------------------------------------------------------------------------
def worker():
    from vllm import LLM, SamplingParams

    deadline = float(os.environ["DEADLINE"])
    pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"), columns=["id", "instruction", "output"])
    pool = pool.sample(frac=1, random_state=SEED).reset_index(drop=True)
    done = find_previous()
    todo = pool[~pool["id"].isin(set(done["id"]))]
    print(f"[judge] {len(pool):,} rows, {len(done):,} already scored, {len(todo):,} to do", flush=True)
    if todo.empty:
        return

    llm = LLM(model=JUDGE_MODEL, dtype="float16", tensor_parallel_size=2, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, seed=SEED, enforce_eager=os.environ.get("VLLM_EAGER") == "1")
    tok = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    chunk_s = CHUNK / 4_374 * 3600                  # Stage 3a rate until the first chunk is timed
    for s in range(0, len(todo), CHUNK):
        if time.time() + chunk_s > deadline:
            print(f"[judge] time budget reached, stopping at {len(done):,} rows", flush=True)
            break
        t = time.time()
        part = todo.iloc[s:s + CHUNK]
        prompts = [tok.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True)
                   for kind in ("complexity", "quality") for m in judge_messages(part, kind)]
        dists = []
        for o in llm.generate(prompts, params, use_tqdm=False):
            p = defaultdict(float)
            for tid, lp in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).items():
                tok_str = lp.decoded_token if getattr(lp, "decoded_token", None) is not None else tok.decode([tid])
                if tok_str.strip() in DIGITS:
                    p[tok_str.strip()] += math.exp(lp.logprob)
            dists.append(p)
        c, c_mass = expectation(dists[:len(part)])
        q, q_mass = expectation(dists[len(part):])
        new = pd.DataFrame({"id": part["id"].values, "complexity_en": c, "complexity_en_mass": c_mass,
                            "quality_en": q, "quality_en_mass": q_mass})
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(SCORES, index=False)
        chunk_s = time.time() - t
        print(f"[judge] {len(done):,} / {len(pool):,}  chunk {chunk_s / 60:.1f} min "
              f"({len(part) / chunk_s * 3600:,.0f} rows/h)", flush=True)


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
def main():
    T0 = time.time()
    deadline = T0 + TIME_BUDGET_H * 3600
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("no HF_TOKEN secret (Qwen is ungated, only rate limits differ):", type(e).__name__)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"])
    # vllm upgrades torch (2.13+cu130) but leaves Kaggle's cu128 torchaudio, which transformers then imports and
    # crashes on. vllm does not need torchaudio, so remove it (Stage 3a v2).
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    subprocess.run([sys.executable, "-c", "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, "
                    "torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"])
    print(f"install done in {(time.time() - T0) / 60:.1f} min", flush=True)

    n_before = len(find_previous())
    t_score = time.time()
    for attempt, extra in enumerate(({}, {"VLLM_EAGER": "1"}, {"VLLM_EAGER": "1"})):
        if time.time() + 15 * 60 > deadline:
            break
        env = {**os.environ, **extra, "DEADLINE": str(deadline), "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        try:        # a hung worker must not run into the 12 h kill, which loses the whole output
            code = subprocess.run([sys.executable, __file__, "worker"], env=env,
                                  timeout=deadline - time.time() + 30 * 60).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        print(f"--- worker attempt {attempt} {extra}: exit {code}", flush=True)
        if code == 0:
            break

    pool = pd.read_parquet(find_input("full/pool/pool_clean.parquet"), columns=["id", "domain"])
    done = find_previous()
    m = pool.merge(done, on="id", how="inner")
    ok = (m["complexity_en_mass"] > JUDGE_MIN_MASS) & (m["quality_en_mass"] > JUDGE_MIN_MASS)
    score_h = (time.time() - t_score) / 3600
    summary = {
        "pool_rows": len(pool), "scored": len(m), "remaining": len(pool) - len(m), "complete": len(m) == len(pool),
        "scored_this_session": len(m) - n_before,
        "rows_per_h_this_session": round((len(m) - n_before) / score_h) if score_h > 0 else None,
        "parse_rate": round(float(ok.mean()), 4) if len(m) else None,
        "median_c_by_domain": m.groupby("domain")["complexity_en"].median().round(2).to_dict(),
        "median_q_by_domain": m.groupby("domain")["quality_en"].median().round(2).to_dict(),
        "total_minutes": round((time.time() - T0) / 60, 1),
    }
    (OUT / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        print(f"INCOMPLETE: {summary['remaining']:,} rows left; push the next bn-full-s03-judge-NN with this output attached")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
    else:
        main()
