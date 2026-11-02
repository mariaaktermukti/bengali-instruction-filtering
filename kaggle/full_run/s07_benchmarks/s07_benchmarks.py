"""Full run, Stage 7 (GPU, Kaggle T4 x2): downstream benchmarks for the base model and every Stage 5 adapter.

  * mmlu: Bengali MMLU (openai/MMMLU, 14,042 questions, 57 subjects). Deterministic: the prompt ends in "সঠিক উত্তর:" and
    the model's next-token log-probs for A / B / C / D decide. All 17 models.
  * rqa: BanglaRQA test set (1,493 questions, native Bengali). Greedy answer after the prefill "উত্তর:", because the trained
    arms otherwise open with a step-by-step preamble and F1 would measure style. The first line is the answer: F1 against
    the best gold span; an unanswerable question scores 1 if that line says "উত্তর নেই". All 17 models. The raw responses
    are saved, so scoring can be redone without a GPU.
  * gen: greedy responses on the 470 held-out prompts, the input of the blind pairwise judging (Stage 8) and the source of
    the generation samples for the paper. Base + seed 42 of each arm (arm A has only seed 42).

Stage 1 decontaminated the pool against both benchmarks. Precision is as in Stage 2b: fp32 weights + fp16 autocast, because
Qwen2.5 in pure fp16 can overflow. Before scoring, each model must reproduce its Stage 5 held-out losses on 3 rows, so an
adapter that silently failed to load stops the run. Runs are claimed from a shared queue, heaviest first, one model per GPU.

Kaggle: GPU T4 x2, Internet ON, dataset habiburrahman118/bn-full-arms, kernel output habiburrahman118/bn-full-s05-lora-04.
Attach an earlier bn-full-s07-benchmarks-NN output to resume; finished files are copied over and skipped.
Outputs under /kaggle/working/bangla_filtering/full/benchmarks/:
    mmlu/<run>.csv, rqa/<run>.jsonl, generations/<run>.jsonl
    mmlu_per_run.csv, mmlu_comparison.csv, mmlu_by_subject.csv
    rqa_per_run.csv, rqa_comparison.csv, rqa_by_type.csv
    generation_stats.csv, stage7_summary.json
"""
import os, sys, glob, json, re, time, shutil, subprocess, unicodedata
from pathlib import Path

INPUT_ROOT = os.environ.get("S07_INPUT", "/kaggle/input")
OUT = Path(os.environ.get("S07_OUT", "/kaggle/working/bangla_filtering/full/benchmarks"))
SMOKE = os.environ.get("S07_SMOKE") == "1"      # 40 MMLU, 10 RQA, 4 held-out rows: tests the code on a CPU or a short session
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "6"))
MIN_START_MIN = 30                              # do not start a new task with less time than this left

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MMLU_REPO, MMLU_REV = "openai/MMMLU", "325a01dc3e173cac1578df94120499aaca2e2504"
RQA_REPO, RQA_REV = "sartajekram/BanglaRQA", "f55f490265170e69073ae2d62c9c79ff405be6e6"
GEN_SEED = "seed42"

MMLU_BATCH_TOKENS = 24_000                      # batch size x longest prompt
RQA_BATCH_TOKENS = 24_000
GEN_BATCH_TOKENS = 48_000                       # batch size x (longest prompt + new tokens)
MAX_BATCH = 64
RQA_NEW_TOKENS = 128                            # ~110 characters; gold answers: p50 18, p90 98
GEN_NEW_TOKENS = 64 if SMOKE else 1536          # held-out responses in characters: p50 821, p90 1,807, p99.9 3,854
REPETITION_PENALTY = 1.0                        # plain greedy: the Qwen default 1.1 would penalise byte-level pieces of Bengali
N_BOOTSTRAP = 10_000
SEED = 42
ABSTAIN = "উত্তর নেই"
MMLU_PROMPT = ("নিচের বহুনির্বাচনী প্রশ্নের সঠিক উত্তরের অক্ষরটি (A, B, C অথবা D) লিখুন।\n\n"
               "প্রশ্ন: {Question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}")
MMLU_PREFILL = "সঠিক উত্তর:"
RQA_PROMPT = (f"নিচের অনুচ্ছেদ পড়ে প্রশ্নের সংক্ষিপ্ত উত্তর দিন, অনুচ্ছেদের শব্দ ব্যবহার করে। "
              f"অনুচ্ছেদে উত্তর না থাকলে শুধু লিখুন: {ABSTAIN}\n\nঅনুচ্ছেদ: {{context}}\n\nপ্রশ্ন: {{question}}")
RQA_PREFILL = "উত্তর:"


def find_file(name):
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not under {INPUT_ROOT}")
    return hits[0]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def lora_root():
    hits = sorted(glob.glob(f"{INPUT_ROOT}/**/full/lora", recursive=True))
    if not hits:
        raise SystemExit("Attach the completed Stage 5 kernel as an input.")
    return Path(hits[0])


def complete_runs(root):
    runs = []
    for d in sorted(root.glob("arm_*_seed*")):
        report = json.loads((d / "training_report.json").read_text(encoding="utf-8"))
        if report["complete"]:
            runs.append(d.name)
    return runs


def tasks_of(run):
    return ["mmlu", "rqa"] + (["gen"] if run == "base" or run.endswith(GEN_SEED) else [])


def task_path(task, run):
    return {"mmlu": OUT / "mmlu" / f"{run}.csv", "rqa": OUT / "rqa" / f"{run}.jsonl",
            "gen": OUT / "generations" / f"{run}.jsonl"}[task]


def arm_of(run):
    return "base" if run == "base" else run.split("_")[1]


def load_mmlu():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    df = pd.read_csv(hf_hub_download(MMLU_REPO, "test/mmlu_BN-BD.csv", repo_type="dataset", revision=MMLU_REV),
                     keep_default_na=False, dtype=str)
    rows = [{"id": i, **{k: r[k] for k in ("Question", "A", "B", "C", "D", "Answer", "Subject")}}
            for i, r in enumerate(df.to_dict("records"))]
    return rows[:40] if SMOKE else rows


def load_rqa():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(RQA_REPO, "Test.json", repo_type="dataset", revision=RQA_REV)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)["data"]
    rows = [{"id": q["question_id"], "context": p["context"], "question": q["question_text"],
             "answerable": q["is_answerable"] == "1", "type": q["question_type"],
             "golds": [g for g in q["answers"]["answer_text"] if g.strip()]}
            for p in data for q in p["qas"]]
    return rows[:10] if SMOKE else rows


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "").lower()
    text = "".join(ch if unicodedata.category(ch)[0] not in "PS" else " " for ch in text)
    return " ".join(text.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    remaining, common = list(g), 0
    for token in p:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def score_rqa(text, row):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    answer = lines[0] if lines else ""
    abstain = normalize(ABSTAIN) in normalize(answer)
    if row["answerable"]:
        best = 0.0 if abstain else max((f1(answer, g) for g in row["golds"]), default=0.0)
        contains = (not abstain) and any(normalize(g) in normalize(text) for g in row["golds"])
    else:
        best, contains = float(abstain), abstain
    return {"f1": round(best, 4), "contains": contains, "abstain": abstain}


# ----------------------------------------------------------------------------------------------------
# worker: python s07_benchmarks.py worker <gpu>   (CUDA_VISIBLE_DEVICES set by the parent)
# ----------------------------------------------------------------------------------------------------
def worker(gpu, deadline):
    import pandas as pd
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = lambda: torch.autocast(device, dtype=torch.float16, enabled=device == "cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad = tok.pad_token_id
    stop_ids = {tok.eos_token_id, pad, tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|im_end|>")}
    letters = [tok(f" {x}", add_special_tokens=False)["input_ids"] for x in "ABCD"]
    assert all(len(x) == 1 for x in letters), letters
    letters = [x[0] for x in letters]

    def prompt_ids(user, prefill=""):
        text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return tok(text + prefill, add_special_tokens=False)["input_ids"]

    def left_pad(seqs):
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, width - len(s):] = torch.tensor(s)
            att[i, width - len(s):] = 1
        return ids.to(device), att.to(device)

    def run_batched(seqs, budget, reserve, fn):
        """Longest first, so an out-of-memory error shows up early; a batch that runs out of memory is split in two."""
        out = [None] * len(seqs)

        def run(idx):
            try:
                for i, result in zip(idx, fn([seqs[i] for i in idx])):
                    out[i] = result
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                print(f"[gpu {gpu}] OOM at batch {len(idx)}, splitting", flush=True)
                run(idx[:len(idx) // 2])
                run(idx[len(idx) // 2:])

        batch = []
        for i in sorted(range(len(seqs)), key=lambda k: -len(seqs[k])):
            longest = len(seqs[batch[0]]) if batch else len(seqs[i])
            if batch and (len(batch) >= MAX_BATCH or (len(batch) + 1) * (longest + reserve) > budget):
                run(batch)
                batch = []
            batch.append(i)
        if batch:
            run(batch)
        return out

    @torch.no_grad()
    def letter_logprobs(model, batch):
        ids, att = left_pad(batch)
        position_ids = (att.cumsum(1) - 1).clamp(min=0)     # as generate() does for left-padded input
        with amp():
            logits = model(input_ids=ids, attention_mask=att, position_ids=position_ids, use_cache=False,
                           logits_to_keep=1).logits[:, -1]
        return logits[:, letters].float().log_softmax(-1).cpu().tolist()

    @torch.no_grad()
    def greedy(model, batch, max_new_tokens):
        ids, att = left_pad(batch)
        with amp():
            out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, repetition_penalty=REPETITION_PENALTY,
                                 pad_token_id=pad, use_cache=True)
        results = []
        for seq in out[:, ids.shape[1]:].cpu().tolist():
            end = next((i for i, t in enumerate(seq) if t in stop_ids), None)
            kept = seq if end is None else seq[:end]
            results.append({"text": tok.decode(kept, skip_special_tokens=True), "tokens": len(kept), "hit_limit": end is None})
        return results

    @torch.no_grad()
    def check_heldout(model, run, rows, expected):
        """Same computation as Stage 5 (no autocast, fp32); the merged weights must give the same losses."""
        for r in rows:
            user = [{"role": "user", "content": r["instruction"]}]
            prompt_text = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full_text = tok.apply_chat_template(user + [{"role": "assistant", "content": r["output"]}], tokenize=False)
            p = tok(prompt_text, add_special_tokens=False)["input_ids"]
            a = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
            ids = torch.tensor([(p + a)[:3072]], device=device)
            labels = torch.tensor([([-100] * len(p) + a)[:3072]], device=device)
            loss = model(input_ids=ids, labels=labels).loss.item()
            if abs(loss - expected[r["id"]]) > 0.01:
                raise RuntimeError(f"{run}: held-out loss {loss:.4f} on {r['id']}, Stage 5 wrote {expected[r['id']]:.4f}")

    def run_mmlu(model):
        rows = load_mmlu()
        seqs = [prompt_ids(MMLU_PROMPT.format(**r), MMLU_PREFILL) for r in rows]
        scores = run_batched(seqs, MMLU_BATCH_TOKENS, 0, lambda b: letter_logprobs(model, b))
        records = []
        for r, s in zip(rows, scores):
            if any(x != x for x in s):
                raise RuntimeError(f"NaN log-prob on MMLU question {r['id']}")
            gold = "ABCD".index(r["Answer"])
            pred = max(range(4), key=s.__getitem__)
            records.append({"id": r["id"], "subject": r["Subject"], "gold": r["Answer"], "pred": "ABCD"[pred],
                            "correct": int(pred == gold), "gold_logprob": round(s[gold], 5)})
        return pd.DataFrame(records).to_csv(index=False, lineterminator="\n")

    def run_rqa(model):
        rows = load_rqa()
        seqs = [prompt_ids(RQA_PROMPT.format(context=r["context"], question=r["question"]), RQA_PREFILL) for r in rows]
        answers = run_batched(seqs, RQA_BATCH_TOKENS, RQA_NEW_TOKENS, lambda b: greedy(model, b, RQA_NEW_TOKENS))
        return [{"id": r["id"], "type": r["type"], "answerable": r["answerable"], "response": a["text"],
                 "tokens": a["tokens"], "hit_limit": a["hit_limit"], **score_rqa(a["text"], r)} for r, a in zip(rows, answers)]

    def run_gen(model, heldout):
        seqs = [prompt_ids(h["instruction"]) for h in heldout]
        answers = run_batched(seqs, GEN_BATCH_TOKENS, GEN_NEW_TOKENS, lambda b: greedy(model, b, GEN_NEW_TOKENS))
        return [{"id": h["id"], "domain": h["domain"], "instruction": h["instruction"], "response": a["text"],
                 "tokens": a["tokens"], "hit_limit": a["hit_limit"]} for h, a in zip(heldout, answers)]

    heldout = load_jsonl(find_file("heldout.jsonl"))
    root = lora_root()
    check_rows = sorted(heldout, key=lambda h: len(h["output"]))[:3]
    heldout = heldout[:4] if SMOKE else heldout
    claims = OUT / ".claims"
    queue = json.loads(os.environ["S07_RUNS"])

    for run in queue:
        todo = [t for t in tasks_of(run) if not task_path(t, run).exists()]
        if not todo:
            continue
        try:
            os.close(os.open(claims / run, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        if time.time() > deadline - MIN_START_MIN * 60:
            print(f"[gpu {gpu}] {run}: not enough time left, not started", flush=True)
            continue
        t0 = time.time()
        # .float(): fp32 master weights; new transformers would load Qwen in bf16, which a T4 only emulates
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).float().to(device)
        if run != "base":
            model = PeftModel.from_pretrained(model, str(root / run)).merge_and_unload().to(device)
        model.eval()
        csv_path = root / "base_heldout_per_example.csv" if run == "base" else root / run / "heldout_per_example.csv"
        check_heldout(model, run, check_rows, dict(pd.read_csv(csv_path)[["id", "loss"]].itertuples(index=False)))
        print(f"[gpu {gpu}] {run}: held-out check passed", flush=True)
        for task in todo:
            if time.time() > deadline - MIN_START_MIN * 60:
                print(f"[gpu {gpu}] {run}/{task}: not enough time left, not started", flush=True)
                continue
            t = time.time()
            if task == "mmlu":
                write_atomic(task_path(task, run), run_mmlu(model))
            elif task == "rqa":
                write_jsonl(task_path(task, run), run_rqa(model))
            else:
                write_jsonl(task_path(task, run), run_gen(model, heldout))
            print(f"[gpu {gpu}] {run}/{task}: {(time.time() - t) / 60:.1f} min", flush=True)
        print(f"[gpu {gpu}] {run}: done, {(time.time() - t0) / 60:.1f} min, peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else 0:.2f} GiB", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
def paired_bootstrap(vectors, value_name):
    """Paired over questions, as in Stage 6; a positive difference means better than the reference here."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(SEED)
    rows = []
    for model, values in vectors.items():
        row = {"model": model, "n": len(values), value_name: round(float(values.mean()), 4)}
        for ref in ("base", "B"):
            if model == ref or ref not in vectors:
                continue
            diff = values - vectors[ref]
            samples = rng.integers(0, len(diff), size=(N_BOOTSTRAP, len(diff)))
            boot = diff[samples].mean(axis=1)
            row[f"diff_vs_{ref}"] = round(float(diff.mean()), 4)
            row[f"ci95_low_vs_{ref}"] = round(float(np.quantile(boot, 0.025)), 4)
            row[f"ci95_high_vs_{ref}"] = round(float(np.quantile(boot, 0.975)), 4)
            row[f"p_higher_vs_{ref}"] = round(float((boot > 0).mean()), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def arm_vectors(frames, value):
    """Mean over an arm's seeds per question, then one vector per arm on a shared question order."""
    import pandas as pd
    wide = pd.concat([f[["id", value]].assign(model=arm_of(run)) for run, f in frames.items()])
    table = wide.groupby(["model", "id"])[value].mean().unstack("model")
    if table.isna().any().any():
        raise SystemExit(f"{value}: the runs do not cover the same questions")
    return {model: table[model].to_numpy(dtype=float) for model in table.columns}


def summarize():
    import pandas as pd
    summary = {"smoke": SMOKE}

    mmlu = {p.stem: pd.read_csv(p) for p in sorted((OUT / "mmlu").glob("*.csv"))}
    if mmlu:
        per_run = [{"run": run, "n": len(f), "accuracy": round(f["correct"].mean(), 4),
                    "macro_subject_accuracy": round(f.groupby("subject")["correct"].mean().mean(), 4),
                    "mean_gold_logprob": round(f["gold_logprob"].mean(), 4)} for run, f in mmlu.items()]
        pd.DataFrame(per_run).to_csv(OUT / "mmlu_per_run.csv", index=False)
        comparison = paired_bootstrap(arm_vectors(mmlu, "correct"), "accuracy")
        comparison.to_csv(OUT / "mmlu_comparison.csv", index=False)
        by_subject = (pd.concat([f.assign(model=arm_of(run)) for run, f in mmlu.items()])
                      .groupby(["model", "subject"])["correct"].agg(["size", "mean"]).round(4).reset_index()
                      .rename(columns={"size": "n", "mean": "accuracy"}))
        by_subject.to_csv(OUT / "mmlu_by_subject.csv", index=False)
        print(comparison.to_string(index=False), flush=True)
        summary["mmlu_runs"] = len(mmlu)

    rqa = {p.stem: pd.DataFrame(load_jsonl(p)) for p in sorted((OUT / "rqa").glob("*.jsonl"))}
    if rqa:
        per_run = []
        for run, f in rqa.items():
            answerable, unanswerable = f[f["answerable"]], f[~f["answerable"]]
            per_run.append({"run": run, "n": len(f), "f1_all": round(f["f1"].mean(), 4),
                            "f1_answerable": round(answerable["f1"].mean(), 4),
                            "contains_answerable": round(answerable["contains"].mean(), 4),
                            "abstain_answerable": round(answerable["abstain"].mean(), 4),
                            "abstain_unanswerable": round(unanswerable["abstain"].mean(), 4),
                            "hit_limit": round(f["hit_limit"].mean(), 4)})
        pd.DataFrame(per_run).to_csv(OUT / "rqa_per_run.csv", index=False)
        comparison = paired_bootstrap(arm_vectors(rqa, "f1"), "f1_all")
        comparison.to_csv(OUT / "rqa_comparison.csv", index=False)
        by_type = (pd.concat([f.assign(model=arm_of(run)) for run, f in rqa.items()])
                   .groupby(["model", "type", "answerable"])["f1"].agg(["size", "mean"]).round(4).reset_index()
                   .rename(columns={"size": "n", "mean": "f1"}))
        by_type.to_csv(OUT / "rqa_by_type.csv", index=False)
        print(comparison.to_string(index=False), flush=True)
        summary["rqa_runs"] = len(rqa)

    bengali = re.compile(r"[ঀ-৿]")
    stats = []
    for p in sorted((OUT / "generations").glob("*.jsonl")):
        rows = load_jsonl(p)
        text = "".join(r["response"] for r in rows)
        stats.append({"run": p.stem, "n": len(rows), "mean_tokens": round(sum(r["tokens"] for r in rows) / len(rows), 1),
                      "hit_limit": round(sum(r["hit_limit"] for r in rows) / len(rows), 4),
                      "empty": sum(not r["response"].strip() for r in rows),
                      "bengali_char_share": round(len(bengali.findall(text)) / max(len(text), 1), 4)})
    if stats:
        pd.DataFrame(stats).to_csv(OUT / "generation_stats.csv", index=False)
        print(pd.DataFrame(stats).to_string(index=False), flush=True)
        summary["generation_runs"] = len(stats)
    return summary


def main():
    T0 = time.time()
    deadline = T0 + TIME_BUDGET_H * 3600
    OUT.mkdir(parents=True, exist_ok=True)

    # Kaggle ships torchao 0.10; recent peft refuses to build LoRA layers next to torchao < 0.16 (pilot fix).
    from importlib.metadata import PackageNotFoundError, version
    try:
        tv = version("torchao")
    except PackageNotFoundError:
        tv = None
    if tv and tuple(int(x) for x in tv.split(".")[:2]) < (0, 16):
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    try:
        import peft  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

    for sub in ("mmlu", "rqa", "generations"):
        (OUT / sub).mkdir(exist_ok=True)
        for f in glob.glob(f"{INPUT_ROOT}/**/full/benchmarks/{sub}/*", recursive=True):
            if not (OUT / sub / Path(f).name).exists():
                shutil.copy(f, OUT / sub)
    claims = OUT / ".claims"
    shutil.rmtree(claims, ignore_errors=True)
    claims.mkdir()

    # warm the caches once, so the workers do not race on the downloads
    load_mmlu(); load_rqa()
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME)

    root = lora_root()
    runs = ["base"] + complete_runs(root)
    if SMOKE:
        runs = [r for r in runs if r in ("base", "arm_B_seed42")]
    elif len(runs) != 17:
        raise SystemExit(f"Expected base + 16 complete adapters, found {runs}")
    queue = sorted(runs, key=lambda r: -len(tasks_of(r)))       # the expensive generation runs first

    import torch
    n_gpu = torch.cuda.device_count()
    print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], "| runs:", queue, flush=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "DEADLINE": str(deadline), "S07_RUNS": json.dumps(queue)}
    workers = [subprocess.Popen([sys.executable, __file__, "worker", str(g)], env={**env, "CUDA_VISIBLE_DEVICES": str(g)})
               for g in range(max(n_gpu, 1))]
    codes = [w.wait() for w in workers]
    print("workers exited with", codes, flush=True)
    shutil.rmtree(claims, ignore_errors=True)

    summary = summarize()
    missing = [f"{run}/{t}" for run in runs for t in tasks_of(run) if not task_path(t, run).exists()]
    summary.update({"runs": runs, "missing": missing, "worker_exit_codes": codes, "model": MODEL_NAME,
                    "mmlu_revision": MMLU_REV, "rqa_revision": RQA_REV, "gen_new_tokens": GEN_NEW_TOKENS,
                    "repetition_penalty": REPETITION_PENALTY, "total_minutes": round((time.time() - T0) / 60, 1)})
    (OUT / "stage7_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if missing:
        print("INCOMPLETE:", missing, "-> push the next bn-full-s07-benchmarks-NN with this output attached")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), float(os.environ["DEADLINE"]))
    else:
        main()
