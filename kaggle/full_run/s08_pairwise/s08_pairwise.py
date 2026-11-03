"""Full run, Stage 8 (GPU, Kaggle T4 x2): blind pairwise judging of held-out generations.

For each of the 470 held-out prompts, every unique pair of models (21 pairs from
base / A / B / C / D / E / F) is compared in both orders (42 comparisons per prompt,
19,740 total). The judge is Qwen2.5-7B-Instruct fp16 via vLLM TP=2 (same backend as
Stage 3, validated at 4,374 rows/h). Model labels are hidden from the prompt (responses
are shown as "Response 1" / "Response 2").

The verdict is extracted from the next-token distribution over tokens "1", "2", and
"T" (tie).  The full logprob distribution is saved so the threshold can be changed
without re-running.

Outputs under /kaggle/working/bangla_filtering/full/pairwise/:
    comparisons.jsonl     all scored rows (id, model_a, model_b, order, verdict, ...)
    pairwise_summary.json run stats, win-rate matrix, resume state

Checkpointing: results are flushed every CHUNK rows. If the session ends before
all comparisons are scored ("missing" is non-empty), push bn-full-s08-pairwise-02 with
this output as an additional kernel_source; the resume block skips finished IDs.

Kaggle: GPU T4 x2, Internet ON, kernel_sources = [habiburrahman118/bn-full-s07-benchmarks-02].
"""
import os, sys, glob, json, math, time, itertools, subprocess
from pathlib import Path
from collections import defaultdict

INPUT_ROOT = os.environ.get("S08_INPUT", "/kaggle/input")
OUT = Path(os.environ.get("S08_OUT", "/kaggle/working/bangla_filtering/full/pairwise"))
SMOKE = os.environ.get("S08_SMOKE") == "1"          # 2 pairs x 5 prompts: tests the code cheaply
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "5.5"))  # stop before the 6h session limit
MIN_START_MIN = 10                                   # do not start a new chunk with less than this left

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VLLM_VERSION = "0.32.0"                             # Stage 3a validated version
MAX_MODEL_LEN = 8_192
SEED = 42
CHUNK = 500                                          # comparisons per checkpoint flush

# Truncation: keep most of the response but not so long that a single comparison hits the
# 8K context limit. Qwen tokenises Bengali at ~1.5-2.5 tokens/char; a 1 500-char response
# is ~700-1000 tokens; two responses + instruction + template ≈ 2 000-3 500 tokens total.
INSTR_CHARS = 1_200
RESP_CHARS = 1_500

VERDICT_TOKENS = {"1": "1", "2": "2", "T": "T", "tie": "T", "Tie": "T", "TIE": "T"}

PAIRWISE_TEMPLATE = """You are evaluating Bengali language model outputs. Given an instruction and two responses, decide which response is better.

Instruction:
<<<
{instruction}
>>>

Response 1:
<<<
{response_a}
>>>

Response 2:
<<<
{response_b}
>>>

Which response is better for this Bengali instruction?
- Reply "1" if Response 1 is clearly better.
- Reply "2" if Response 2 is clearly better.
- Reply "T" if they are roughly tied or you cannot decide.

Reply with a single character: 1, 2, or T."""

MODELS = ["base", "A", "B", "C", "D", "E", "F"]


def find_gen_dir():
    """Find the generations/ directory from the attached Stage 7 output."""
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/benchmarks/generations", recursive=True))
    if not hits:
        raise SystemExit("No generations/ dir found. Attach bn-full-s07-benchmarks-02 output.")
    return Path(hits[0])


def load_generations(gen_dir):
    """Load all 7 model generation files. Returns {model_label: [row, ...]}."""
    name_map = {
        "base": "base.jsonl",
        "A": "arm_A_seed42.jsonl",
        "B": "arm_B_seed42.jsonl",
        "C": "arm_C_seed42.jsonl",
        "D": "arm_D_seed42.jsonl",
        "E": "arm_E_seed42.jsonl",
        "F": "arm_F_seed42.jsonl",
    }
    gens = {}
    for label, fname in name_map.items():
        path = gen_dir / fname
        if not path.exists():
            raise SystemExit(f"Missing {path}. Attach bn-full-s07-benchmarks-02 output.")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        gens[label] = rows
    # Verify all files have the same IDs in the same order
    ref_ids = [r["id"] for r in gens["base"]]
    for label, rows in gens.items():
        ids = [r["id"] for r in rows]
        if ids != ref_ids:
            raise SystemExit(f"Model {label} IDs do not match base IDs.")
    return gens, ref_ids


def build_comparisons(gens, ref_ids, done_keys):
    """Build the full shuffled list of (comparison_key, comparison_dict) to score.

    comparison_key = f"{prompt_id}|{model_a}|{model_b}|{order}"  (order = 'ab' or 'ba')
    For each prompt and each unordered pair, we create TWO rows: order ab and order ba.
    """
    import random
    rng = random.Random(SEED)
    pairs = list(itertools.combinations(MODELS, 2))        # 21 pairs
    prompts = ref_ids[:5] if SMOKE else ref_ids            # 5 prompts in smoke mode
    pairs_smoke = pairs[:2] if SMOKE else pairs            # 2 pairs in smoke mode

    rows = []
    for pid_idx, pid in enumerate(prompts):
        base_row = gens["base"][pid_idx]
        instruction = base_row["instruction"][:INSTR_CHARS]
        for ma, mb in pairs_smoke:
            row_a = gens[ma][pid_idx]
            row_b = gens[mb][pid_idx]
            ra = row_a["response"][:RESP_CHARS]
            rb = row_b["response"][:RESP_CHARS]
            for order in ("ab", "ba"):
                key = f"{pid}|{ma}|{mb}|{order}"
                if key in done_keys:
                    continue
                resp1, resp2 = (ra, rb) if order == "ab" else (rb, ra)
                rows.append({
                    "key": key,
                    "id": pid,
                    "domain": base_row["domain"],
                    "model_a": ma,
                    "model_b": mb,
                    "order": order,
                    "instruction": instruction,
                    "response_a": resp1,
                    "response_b": resp2,
                })
    rng.shuffle(rows)
    return rows


def load_previous():
    """Load comparison results from earlier sessions (attached kernel output or this session's file)."""
    results_path = OUT / "comparisons.jsonl"
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/pairwise/comparisons.jsonl", recursive=True))
    if results_path.exists():
        hits.append(str(results_path))
    done = {}
    for p in hits:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["key"]] = row
    print(f"[stage8] Loaded {len(done)} previous comparisons from {len(hits)} file(s).", flush=True)
    return done


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_win_rates(results):
    """Compute per-arm win rates versus every other arm, with position-bias correction.

    For each unordered pair (ma, mb), we have comparisons in order 'ab' and 'ba'.
    We average the two orders to cancel position bias:
      win_ma = 0.5 * P(ma wins as model_1 in ab-order) + 0.5 * P(ma wins as model_2 in ba-order)
    """
    import numpy as np
    # Collect per-pair, per-order verdict distributions
    pair_data = defaultdict(lambda: {"ab": [], "ba": []})
    for key, row in results.items():
        if "verdict_expected" not in row:
            continue
        ma, mb, order = row["model_a"], row["model_b"], row["order"]
        pair_key = (ma, mb)                          # always (ma, mb) regardless of order
        pair_data[pair_key][order].append(row)

    # Win matrix: win_matrix[ma][mb] = fraction of (ma, mb) prompt comparisons where ma wins
    win_matrix = defaultdict(lambda: defaultdict(list))
    for (ma, mb), orders in pair_data.items():
        for prompt_id in set(r["id"] for r in orders.get("ab", [])) & set(r["id"] for r in orders.get("ba", [])):
            ab_rows = [r for r in orders["ab"] if r["id"] == prompt_id]
            ba_rows = [r for r in orders["ba"] if r["id"] == prompt_id]
            if not ab_rows or not ba_rows:
                continue
            # In order ab: response_a=ma, response_b=mb; verdict "1" = ma wins
            ab_verdict = ab_rows[0]["verdict_expected"]  # expected from logprobs: 1=response1, 2=response2
            # In order ba: response_a=mb, response_b=ma; verdict "1" = mb wins = ma loses
            ba_verdict = ba_rows[0]["verdict_expected"]
            # Average over position: ma wins if ab says "1" or ba says "2"
            ma_wins = 0.5 * (1.0 if ab_verdict < 1.5 else 0.0 if ab_verdict > 2.5 else 0.5) + \
                      0.5 * (1.0 if ba_verdict > 1.5 else 0.0 if ba_verdict < 1.5 else 0.5)
            win_matrix[ma][mb].append(ma_wins)
            win_matrix[mb][ma].append(1.0 - ma_wins)

    summary = {}
    for ma in MODELS:
        summary[ma] = {}
        for mb in MODELS:
            if ma == mb:
                continue
            vals = win_matrix[ma][mb]
            if vals:
                summary[ma][mb] = round(float(np.mean(vals)), 4)
    return summary


# ----------------------------------------------------------------------------------------------------
# worker: python s08_pairwise.py worker   (DEADLINE env var, vLLM already installed)
# ----------------------------------------------------------------------------------------------------
def worker(deadline):
    from vllm import LLM, SamplingParams

    # Read the todo list from a temp file (too large to pass through an env var)
    todo_path = Path(os.environ["S08_TODO_FILE"])
    with open(todo_path, encoding="utf-8") as f:
        todo = json.load(f)
    out_path = OUT / "comparisons.jsonl"

    llm = LLM(model=JUDGE_MODEL, dtype="float16", tensor_parallel_size=2,
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90,
              seed=SEED, enforce_eager=os.environ.get("VLLM_EAGER") == "1")
    tok = llm.get_tokenizer()
    # Token IDs for verdicts: "1", "2", "T"
    v_ids = {tok.convert_tokens_to_ids(v): label for v, label in [(" 1", "1"), ("1", "1"),
                                                                     (" 2", "2"), ("2", "2"),
                                                                     (" T", "T"), ("T", "T"),
                                                                     (" t", "T"), ("t", "T")]}
    # SamplingParams: 1 token, get logprobs for top 20 (vLLM max allowed without custom engine flags)
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)

    chunk_estimate = CHUNK / 3500 * 3600             # estimated seconds per chunk at 3500 comp/h

    i = 0
    while i < len(todo):
        if time.time() > deadline - MIN_START_MIN * 60:
            print(f"[stage8-worker] time budget reached at {i}/{len(todo)}, stopping.", flush=True)
            break
        chunk = todo[i:i + CHUNK]
        prompts = []
        for row in chunk:
            msg = PAIRWISE_TEMPLATE.format(
                instruction=row["instruction"],
                response_a=row["response_a"],
                response_b=row["response_b"],
            )
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False, add_generation_prompt=True
            ))
        t = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        new_rows = []
        for row, out in zip(chunk, outputs):
            # Extract verdict from logprob distribution
            lp_map = {}
            for tid, lp in (out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}).items():
                label = v_ids.get(tid)
                if label:
                    lp_map[label] = lp_map.get(label, -1e9)
                    lp_map[label] = max(lp_map[label], lp.logprob)
            # Compute expected value: 1="1" (response_a wins), 2="2" (response_b wins), 1.5="T" (tie)
            probs = {k: math.exp(v) for k, v in lp_map.items()}
            total = sum(probs.values()) or 1.0
            probs = {k: v / total for k, v in probs.items()}
            expected = probs.get("1", 0.0) * 1.0 + probs.get("T", 0.0) * 1.5 + probs.get("2", 0.0) * 2.0
            if not probs:
                expected = float("nan")
            result_row = {
                **{k: row[k] for k in ("key", "id", "domain", "model_a", "model_b", "order")},
                "verdict_expected": round(expected, 4) if probs else None,
                "prob_1": round(probs.get("1", 0.0), 4),
                "prob_T": round(probs.get("T", 0.0), 4),
                "prob_2": round(probs.get("2", 0.0), 4),
                "verdict_raw": max(probs, key=probs.get) if probs else None,
            }
            new_rows.append(result_row)
        append_jsonl(out_path, new_rows)
        elapsed = time.time() - t
        chunk_estimate = elapsed
        rate = len(chunk) / elapsed * 3600
        total_done = i + len(chunk)
        print(f"[stage8-worker] {total_done}/{len(todo)} | chunk {elapsed/60:.1f} min ({rate:,.0f} comp/h)", flush=True)
        i += CHUNK


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
def summarize(results, total_expected):
    import json
    win_rates = compute_win_rates(results)
    n_done = len(results)
    missing_count = total_expected - n_done
    summary = {
        "smoke": SMOKE,
        "total_comparisons": total_expected,
        "scored": n_done,
        "missing": missing_count,
        "complete": missing_count == 0,
        "win_rates_vs_B": {m: win_rates.get(m, {}).get("B") for m in MODELS if m != "B"},
        "win_matrix": {ma: {mb: win_rates.get(ma, {}).get(mb) for mb in MODELS if mb != ma} for ma in MODELS},
        "judge_model": JUDGE_MODEL,
        "vllm_version": VLLM_VERSION,
    }
    return summary


def main():
    T0 = time.time()
    deadline = T0 + TIME_BUDGET_H * 3600
    OUT.mkdir(parents=True, exist_ok=True)

    # Install vLLM (same approach as Stage 3)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"vllm=={VLLM_VERSION}"])
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    subprocess.run([sys.executable, "-c",
                    "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, "
                    "torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"])
    print(f"[stage8] install done in {(time.time()-T0)/60:.1f} min", flush=True)

    # Load generations from attached Stage 7 output
    gen_dir = find_gen_dir()
    print(f"[stage8] loading generations from {gen_dir}", flush=True)
    gens, ref_ids = load_generations(gen_dir)
    print(f"[stage8] {len(ref_ids)} prompts, {len(gens)} models", flush=True)

    # Load previous results (for resuming across sessions)
    done_map = load_previous()
    done_keys = set(done_map.keys())

    # Build comparison list (skipping already-done)
    pairs = list(itertools.combinations(MODELS, 2))
    pairs_used = pairs[:2] if SMOKE else pairs
    prompts_used = ref_ids[:5] if SMOKE else ref_ids
    total_expected = len(prompts_used) * len(pairs_used) * 2   # 2 orders per pair
    todo = build_comparisons(gens, ref_ids, done_keys)
    print(f"[stage8] {len(done_keys)} comparisons already done; {len(todo)} remaining (total {total_expected})", flush=True)

    # Copy previous output file to local OUT if it came from an input
    local_results = OUT / "comparisons.jsonl"
    if done_map and not local_results.exists():
        # Write all previous results to local file so the worker can append
        with open(local_results, "w", encoding="utf-8") as f:
            for row in done_map.values():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[stage8] copied {len(done_map)} previous rows to local file", flush=True)

    if not todo:
        print("[stage8] all comparisons already done, going straight to summary.", flush=True)
    else:
        # Write todo list to a temp file (too large for an env var at 19K rows x 3KB each)
        todo_file = OUT / ".todo.json"
        todo_file.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
        env = {
            **os.environ,
            "DEADLINE": str(deadline),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TOKENIZERS_PARALLELISM": "false",
            "S08_TODO_FILE": str(todo_file),
        }
        for attempt, extra in enumerate(({}, {"VLLM_EAGER": "1"}, {"VLLM_EAGER": "1"})):
            if time.time() + 15 * 60 > deadline:
                break
            env_run = {**env, **extra}
            try:
                code = subprocess.run(
                    [sys.executable, __file__, "worker"],
                    env=env_run,
                    timeout=deadline - time.time() + 30 * 60,
                ).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
            print(f"--- worker attempt {attempt} {extra}: exit {code}", flush=True)
            if code == 0:
                break

    # Reload all results (including newly scored ones)
    all_results = {}
    if local_results.exists():
        with open(local_results, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    all_results[row["key"]] = row
    print(f"[stage8] {len(all_results)} total comparisons scored.", flush=True)

    # Compute summary and win rates
    summary = summarize(all_results, total_expected)
    summary["total_minutes"] = round((time.time() - T0) / 60, 1)
    if summary["missing"] > 0:
        print(f"INCOMPLETE: {summary['missing']} comparisons remain. Push bn-full-s08-pairwise-02 "
              f"with this output as a kernel_source.", flush=True)

    write_atomic(OUT / "pairwise_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    # Print win rates vs B for easy reading
    print("\nWin rates vs Arm B (position-bias corrected):", flush=True)
    for m, wr in summary["win_rates_vs_B"].items():
        print(f"  {m}: {wr}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        deadline = float(os.environ["DEADLINE"])
        worker(deadline)
    else:
        main()
