"""Full run, Stage 9 (GPU, Kaggle T4 x2): judge validation + independent blind spot-check.

Two jobs, both needed for the paper:

**1. Judge vs human (ROADMAP phase 0).** The 100 human-annotated rows were labelled on the
pilot, not on the 100K full run, so only 29 of them are in the clean pool and the rest have
no production judge score. This kernel scores those 71 rows with the *same* Stage 3 judge
(Qwen2.5-7B-Instruct fp16, vLLM TP=2, the notebook rubric, digit-expectation scoring) and
merges them with the production Stage 3 scores of the 29 pool rows -> 100 comparable rows.

**2. Independent blind spot-check (ROADMAP phase 2).** The Stage 4 sheet mixes 50 rows from
arm C and 50 from arm B with the arm hidden. A rating of those rows by the *same* judge that
selected the arms would be circular: arm C was chosen to maximise the judge's complexity x
quality. So the sheet is rated by a second, independent model from a different family
(Llama-3.1-8B-Instruct, ungated `unsloth/` mirror), on a single 1-5 "overall usefulness"
scale that matches the human `rating_1to5` column. The Qwen judge scores the same items too,
so the paper can show the circularity and the independent agreement side by side.

Cost: Qwen 171 rows x 2 prompts + Llama 200 rows x 1 prompt = 742 judge calls, ~5 min of GPU
work after ~10 min of setup and model loading.

Kaggle: GPU T4 x2, Internet ON, dataset_sources = [habiburrahman153/bn-full-eval-validation]
Outputs under /kaggle/working/bangla_filtering/full/validation/:
    judge_human_100.jsonl   every human row: judge scores (complexity, quality, mass) + source
    spotcheck_judge.csv      Qwen judge scores for the 100 blind spot-check items
    spotcheck_llama.csv      the independent Llama-3.1-8B ratings for the same 100 items
    judge_human_summary.json run stats
"""

import os
import sys
import json
import glob
import math
import time
import re
import hashlib
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_ROOT = os.environ.get("S09_INPUT", "/kaggle/input")
OUT = Path(os.environ.get("S09_OUT", "/kaggle/working/bangla_filtering/full/validation"))
OUT.mkdir(parents=True, exist_ok=True)

# the Stage 3 judge (the same one that selected the arms)
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# the independent rater: a different family, so the spot-check is not circular
LLAMA_MODEL = "unsloth/Llama-3.1-8B-Instruct"

VLLM_VERSION = "0.32.0"          # the version Stage 3 / Stage 8 validated
MAX_MODEL_LEN = 8_192
SEED = 42
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "2.0"))

# identical to Stage 3 (s03_judge.py) and the pilot notebook
JUDGE_MAX_INSTR_CHARS, JUDGE_MAX_RESP_CHARS = 1_200, 2_000
JUDGE_MIN_MASS = 0.5
DIGITS = [str(d) for d in range(1, 7)]

# --- BEGIN GENERATED RUBRIC (analysis/extract_rubrics.py) ---
RUBRIC = {
 ("complexity", "en"): """You are rating Bengali (বাংলা) instructions for an instruction-tuning dataset.
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
 ("quality", "en"): """You are rating a Bengali (বাংলা) instruction–response pair for an instruction-tuning dataset.
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
 ("complexity", "bn"): """নিচের বাংলা নির্দেশনাটি কতটা জটিল, ১ থেকে ৬ স্কেলে মূল্যায়ন করুন।
1 = খুব সহজ (একটি তথ্য বা এক শব্দের উত্তর)
2 = সহজ (এক ধাপের কাজ, সাধারণ জ্ঞান)
3 = মাঝারি (কয়েকটি ধাপ বা ছোট ব্যাখ্যা)
4 = জটিল (বহু-ধাপের যুক্তি, একাধিক শর্ত বা বিশেষ জ্ঞান)
5 = কঠিন (গভীর যুক্তি, পরস্পর-নির্ভর শর্ত, দীর্ঘ কাঠামোবদ্ধ উত্তর)
6 = অত্যন্ত কঠিন (বিশেষজ্ঞ-স্তরের বহু-ধাপের সমস্যা)
শুধু কাজটি কতটা কঠিন তা বিচার করুন, লেখার মান নয়।

নির্দেশনা:
<<<
{instruction}
>>>

শুধু 1 থেকে 6 এর মধ্যে একটি সংখ্যা লিখুন।""",
 ("quality", "bn"): """নিচের বাংলা নির্দেশনা ও উত্তরের জোড়াটি দেখে উত্তরের মান ১ থেকে ৬ স্কেলে মূল্যায়ন করুন। বিবেচনা করুন:
- তথ্য ও যুক্তির নির্ভুলতা
- নির্দেশনার প্রতিটি অংশ মানা হয়েছে কিনা (ফরম্যাট, দৈর্ঘ্য, সব উপ-প্রশ্ন)
- উত্তর কতটা সহায়ক ও যথেষ্ট গভীর
- বাংলা ভাষার মান: স্বাভাবিক, সাবলীল, ব্যাকরণগতভাবে শুদ্ধ; যান্ত্রিক অনুবাদের ছাপ (ইংরেজি বাক্যগঠন, অনূদিত না হওয়া ইংরেজি, ভাঙা যুক্তাক্ষর) থাকলে নম্বর কমান
1 = ভুল বা অপ্রাসঙ্গিক   2 = বড় ভুল আছে   3 = আংশিক সঠিক বা দুর্বল লেখা
4 = সঠিক ও গ্রহণযোগ্য   5 = সঠিক, সম্পূর্ণ ও সাবলীল   6 = চমৎকার, দক্ষ বাংলাভাষী বিশেষজ্ঞের মানের

নির্দেশনা:
<<<
{instruction}
>>>

উত্তর:
<<<
{response}
>>>

শুধু 1 থেকে 6 এর মধ্যে একটি সংখ্যা লিখুন।""",
}
# --- END GENERATED RUBRIC ---

# New for Stage 9: a single 1-5 "overall usefulness" rating, so the LLM ratings are on the
# same scale as the human `rating_1to5` column of the blind spot-check sheet.
USEFULNESS_RUBRIC = """You are rating a Bengali (বাংলা) instruction–response pair for an instruction-tuning dataset.
Rate the OVERALL USEFULNESS of the pair from 1 to 5, considering:
- correctness and factual accuracy
- whether it follows every part of the instruction (format, length, all sub-questions)
- helpfulness and appropriate depth
- Bengali language quality: natural, fluent, grammatical; penalise machine-translation artefacts
  (literal English word order, untranslated English, broken conjuncts, wrong script)
1 = wrong, off-topic or unusable   2 = major errors   3 = partly correct or poorly written
4 = correct and acceptable   5 = correct, complete and fluent

Instruction:
<<<
{instruction}
>>>

Response:
<<<
{response}
>>>

Reply with one digit from 1 to 5."""


def clip(s, n):
    return s if len(s) <= n else s[:n]


def clip_middle(s, n):
    """Keep the start and the final answer (math answers sit at the end)."""
    return s if len(s) <= n else s[: int(n * 0.7)] + "\n…\n" + s[-int(n * 0.3):]


def judge_messages(part, kind, lang="en"):
    tmpl = RUBRIC[(kind, lang)]
    return [
        tmpl.format(
            instruction=clip(i, JUDGE_MAX_INSTR_CHARS),
            response=clip_middle(o, JUDGE_MAX_RESP_CHARS),
        )
        for i, o in zip(part["instruction"], part["output"])
    ]


def usefulness_messages(part):
    return [
        USEFULNESS_RUBRIC.format(
            instruction=clip(i, JUDGE_MAX_INSTR_CHARS),
            response=clip_middle(o, JUDGE_MAX_RESP_CHARS),
        )
        for i, o in zip(part["instruction"], part["output"])
    ]


def expectation(dists):
    mass = np.array([sum(p.values()) for p in dists])
    expected = np.array([
        sum(int(d) * v for d, v in p.items()) / m if m > 0 else np.nan
        for p, m in zip(dists, mass)
    ])
    return expected, mass


def find_input(pattern):
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{pattern} not under {INPUT_ROOT}")
    return hits[0]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def loose(s):
    """Punctuation/whitespace-insensitive key (mirrors the Stage 1 dedup key)."""
    return re.sub(r"[\s।॥!-/:-@\[-`{-~“”‘’…–—]+", "",
                  unicodedata.normalize("NFC", str(s)).lower())


def loose_hash(s):
    return hashlib.sha1(loose(s).encode("utf-8")).hexdigest()


def load_human_rows():
    """100 human-annotated rows: id, instruction, response, human_complexity, human_quality."""
    path = find_input("human_annotation_100.jsonl")
    rows = load_jsonl(path)
    for r in rows:
        r["human_complexity"] = int(r["human_complexity"])
        r["human_quality"] = int(r["human_quality"])
    print(f"[stage9] {len(rows)} human rows from {path}", flush=True)
    return rows


def load_spotcheck_rows():
    """The 100 blind spot-check items (50 B + 50 C, arm hidden)."""
    path = find_input("spotcheck_blind.csv")
    df = pd.read_csv(path)
    df["item"] = df["item"].astype(int)
    print(f"[stage9] {len(df)} spot-check rows from {path}", flush=True)
    return df


def load_pool_matches():
    """Pool rows whose instruction matches a human row, with the production Stage 3 scores."""
    path = find_input("pool_matches.csv")
    df = pd.read_csv(path)
    df["loose_instruction"] = df["loose_instruction"].astype(str)
    df["loose_response_hash"] = df["loose_response_hash"].astype(str)
    print(f"[stage9] {len(df)} candidate pool rows from {path}", flush=True)
    return df


def match_to_pool(human_rows, pool):
    """Split the human rows into those with a reusable production score and those that need a fresh pass."""
    by_instr = defaultdict(list)
    for rec in pool.to_dict("records"):
        by_instr[rec["loose_instruction"]].append(rec)

    matched, unmatched = [], []
    for h in human_rows:
        cands = by_instr.get(loose(h["instruction"]), [])
        resp_hash = loose_hash(h["response"])
        exact = [c for c in cands if c["loose_response_hash"] == resp_hash]
        if not exact:
            unmatched.append(h)
            continue
        rec = exact[0]
        h = dict(h)
        h["pool_id"] = rec["id"]
        h["c"] = rec["c"]
        h["q"] = rec["q"]
        h["source"] = "stage3_pool"
        matched.append(h)
    print(f"[stage9] {len(matched)} human rows reuse the production Stage 3 score, "
          f"{len(unmatched)} need a fresh pass", flush=True)
    return matched, unmatched


# ----------------------------------------------------------------------------------------------------
# worker: python s09_judge_human.py worker
#   S09_MODEL      which model to load
#   S09_TODO_FILE  the rows to score
#   S09_OUT_FILE   where to write the scores
#   S09_MODE       "judge" (complexity + quality, Stage 3 rubric) or "usefulness" (1-5, single prompt)
# ----------------------------------------------------------------------------------------------------
def worker():
    from vllm import LLM, SamplingParams

    model_name = os.environ["S09_MODEL"]
    mode = os.environ.get("S09_MODE", "judge")
    todo_path = Path(os.environ["S09_TODO_FILE"])
    out_path = Path(os.environ["S09_OUT_FILE"])
    with open(todo_path, encoding="utf-8") as f:
        todo = json.load(f)

    llm = LLM(model=model_name, dtype="float16", tensor_parallel_size=2,
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90, seed=SEED,
              enforce_eager=os.environ.get("VLLM_EAGER") == "1")
    tok = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    part = pd.DataFrame(todo)
    n = len(part)

    if mode == "judge":
        prompts = []
        for kind in ("complexity", "quality"):
            for m in judge_messages(part, kind):
                prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True))
    else:
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True)
            for m in usefulness_messages(part)]

    dists = []
    for o in llm.generate(prompts, params, use_tqdm=False):
        p = defaultdict(float)
        for tid, lp in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).items():
            tok_str = lp.decoded_token if getattr(lp, "decoded_token", None) is not None else tok.decode([tid])
            if tok_str.strip() in DIGITS:
                p[tok_str.strip()] += math.exp(lp.logprob)
        dists.append(p)

    rows = []
    if mode == "judge":
        c, c_mass = expectation(dists[:n])
        q, q_mass = expectation(dists[n:])
        for i, rec in enumerate(part.to_dict("records")):
            rows.append({
                "id": rec["id"],
                "complexity_en": float(c[i]),
                "complexity_en_mass": float(c_mass[i]),
                "quality_en": float(q[i]),
                "quality_en_mass": float(q_mass[i]),
            })
    else:
        u, u_mass = expectation(dists)
        for i, rec in enumerate(part.to_dict("records")):
            rows.append({
                "id": rec["id"],
                "usefulness": float(u[i]),
                "usefulness_mass": float(u_mass[i]),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[stage9-worker] {model_name} ({mode}) scored {len(rows)} rows -> {out_path}", flush=True)


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
def run_worker(model_name, mode, todo, deadline, out_file):
    """Score `todo` with `model_name` in a fresh process (one vLLM instance per model)."""
    todo_file = OUT / f".todo_{mode}.json"
    todo_file.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
    env = {
        **os.environ,
        "DEADLINE": str(deadline),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "S09_MODEL": model_name,
        "S09_MODE": mode,
        "S09_TODO_FILE": str(todo_file),
        "S09_OUT_FILE": str(out_file),
    }
    for attempt, extra in enumerate(({}, {"VLLM_EAGER": "1"}, {"VLLM_EAGER": "1"})):
        if time.time() + 10 * 60 > deadline:
            print(f"[stage9] time budget reached before {model_name}", flush=True)
            return False
        env_run = {**env, **extra}
        try:
            code = subprocess.run([sys.executable, __file__, "worker"], env=env_run,
                                  timeout=deadline - time.time() + 20 * 60).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        print(f"--- worker {model_name} ({mode}) attempt {attempt} {extra}: exit {code}", flush=True)
        if code == 0:
            return True
    return False


def main():
    T0 = time.time()
    deadline = T0 + TIME_BUDGET_H * 3600
    OUT.mkdir(parents=True, exist_ok=True)

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"])
    # vllm upgrades torch but leaves Kaggle's cu128 torchaudio, which transformers then imports
    # and crashes on. vLLM does not need torchaudio (Stage 3a v2 / Stage 8).
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    subprocess.run([sys.executable, "-c",
                    "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, "
                    "torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"])
    print(f"[stage9] install done in {(time.time() - T0) / 60:.1f} min", flush=True)

    human_rows = load_human_rows()
    spot_rows = load_spotcheck_rows()
    pool = load_pool_matches()

    matched, unmatched = match_to_pool(human_rows, pool)

    # ---- pass 1: the Stage 3 judge on the rows it has not scored yet --------------------------
    qwen_todo = []
    for h in unmatched:
        qwen_todo.append({
            "id": f"human_{h['id']}",
            "instruction": h["instruction"],
            "output": h["response"],
        })
    for rec in spot_rows.to_dict("records"):
        qwen_todo.append({
            "id": f"spot_{int(rec['item'])}",
            "instruction": str(rec["instruction"]),
            "output": str(rec["output"]),
        })
    print(f"[stage9] pass 1: {len(qwen_todo)} rows for the Stage 3 judge "
          f"({len(unmatched)} human + {len(spot_rows)} spot-check)", flush=True)

    qwen_out = OUT / "fresh_scores.jsonl"
    qwen_ok = run_worker(QWEN_MODEL, "judge", qwen_todo, deadline, qwen_out)
    fresh = load_jsonl(qwen_out) if qwen_out.exists() else []
    fresh_by_id = {r["id"]: r for r in fresh}
    print(f"[stage9] pass 1 produced {len(fresh)} scores (ok={qwen_ok})", flush=True)

    # ---- pass 2: the independent rater ------------------------------------------------------
    llama_todo = []
    for h in human_rows:
        llama_todo.append({
            "id": f"human_{h['id']}",
            "instruction": h["instruction"],
            "output": h["response"],
        })
    for rec in spot_rows.to_dict("records"):
        llama_todo.append({
            "id": f"spot_{int(rec['item'])}",
            "instruction": str(rec["instruction"]),
            "output": str(rec["output"]),
        })
    print(f"[stage9] pass 2: {len(llama_todo)} rows for the independent rater", flush=True)

    llama_out = OUT / "llama_scores.jsonl"
    llama_ok = run_worker(LLAMA_MODEL, "usefulness", llama_todo, deadline, llama_out)
    llama_rows = load_jsonl(llama_out) if llama_out.exists() else []
    llama_by_id = {r["id"]: r for r in llama_rows}
    print(f"[stage9] pass 2 produced {len(llama_rows)} scores (ok={llama_ok})", flush=True)

    # ---- assemble the human validation table -------------------------------------------------
    pool_by_hash = defaultdict(list)
    for rec in pool.to_dict("records"):
        pool_by_hash[(rec["loose_instruction"], rec["loose_response_hash"])].append(rec)

    out_rows = []
    for h in human_rows:
        hid = h["id"]
        fr = fresh_by_id.get(f"human_{hid}")
        if fr is not None:
            c, q = fr["complexity_en"], fr["quality_en"]
            c_mass, q_mass = fr["complexity_en_mass"], fr["quality_en_mass"]
            source = "stage9_fresh"
        else:
            cands = pool_by_hash.get((loose(h["instruction"]), loose_hash(h["response"])), [])
            if not cands:
                continue
            rec = cands[0]
            c, q = rec["c"], rec["q"]
            c_mass, q_mass = np.nan, np.nan
            source = "stage3_pool"
        lr = llama_by_id.get(f"human_{hid}")
        out_rows.append({
            "id": hid,
            "human_complexity": h["human_complexity"],
            "human_quality": h["human_quality"],
            "llm_complexity": c,
            "llm_quality": q,
            "complexity_mass": c_mass,
            "quality_mass": q_mass,
            "llama_usefulness": lr["usefulness"] if lr else np.nan,
            "llama_usefulness_mass": lr["usefulness_mass"] if lr else np.nan,
            "source": source,
        })

    df_human = pd.DataFrame(out_rows)
    df_human.to_json(OUT / "judge_human_100.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"[stage9] wrote judge_human_100.jsonl ({len(df_human)} rows)", flush=True)

    # ---- assemble the spot-check tables ------------------------------------------------------
    sc_rows = []
    for rec in spot_rows.to_dict("records"):
        item = int(rec["item"])
        fr = fresh_by_id.get(f"spot_{item}")
        lr = llama_by_id.get(f"spot_{item}")
        if fr is None:
            continue
        sc_rows.append({
            "item": item,
            "instruction": rec["instruction"],
            "output": rec["output"],
            "llm_complexity": fr["complexity_en"],
            "llm_quality": fr["quality_en"],
            "complexity_mass": fr["complexity_en_mass"],
            "quality_mass": fr["quality_en_mass"],
            "llama_usefulness": lr["usefulness"] if lr else np.nan,
            "llama_usefulness_mass": lr["usefulness_mass"] if lr else np.nan,
        })
    df_sc = pd.DataFrame(sc_rows)
    df_sc.to_csv(OUT / "spotcheck_judge.csv", index=False, encoding="utf-8-sig")
    print(f"[stage9] wrote spotcheck_judge.csv ({len(df_sc)} rows)", flush=True)

    # ---- summary ---------------------------------------------------------------------------
    summary = {
        "human_rows_total": len(human_rows),
        "human_rows_scored": int(len(df_human)),
        "human_rows_from_stage3_pool": int((df_human["source"] == "stage3_pool").sum()) if len(df_human) else 0,
        "human_rows_fresh": int((df_human["source"] == "stage9_fresh").sum()) if len(df_human) else 0,
        "human_rows_with_llama": int(df_human["llama_usefulness"].notna().sum()) if len(df_human) else 0,
        "spotcheck_rows_scored": int(len(df_sc)),
        "spotcheck_rows_with_llama": int(df_sc["llama_usefulness"].notna().sum()) if len(df_sc) else 0,
        "judge_model": QWEN_MODEL,
        "independent_rater": LLAMA_MODEL,
        "vllm_version": VLLM_VERSION,
        "parse_rate_fresh": round(float(((df_human["complexity_mass"] > JUDGE_MIN_MASS).mean())
                                        if len(df_human) else float("nan")), 4),
        "total_minutes": round((time.time() - T0) / 60, 1),
        "complete": (len(df_human) == len(human_rows)
                     and len(df_sc) == len(spot_rows)
                     and bool(llama_ok)),
    }
    (OUT / "judge_human_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if not summary["complete"]:
        print(f"INCOMPLETE: {summary['human_rows_scored']}/{summary['human_rows_total']} human rows, "
              f"{summary['spotcheck_rows_scored']}/{len(spot_rows)} spot-check rows, "
              f"llama_ok={llama_ok}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker()
    else:
        main()
