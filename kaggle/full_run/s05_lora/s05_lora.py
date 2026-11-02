"""Full run, Stage 5 (GPU, Kaggle T4 x2): LoRA on the selection arms, one run per GPU at a time.

Same training as the pilot (scripts/kaggle_lora_training.py): Qwen2.5-0.5B-Instruct, LoRA r 8 / alpha 16 / dropout 0.05
on q, k, v, o; 1 epoch; lr 1e-4; batch 1 x 4 gradient accumulation; max 3,072 tokens; fp32 weights + fp16 AMP;
Qwen chat template, loss on the response only. Changes for the full run:
  * each GPU works through its own queue of (arm, seed) runs (QUEUES below), so one session trains several arms;
  * set_seed(seed) before LoRA init, so seeds differ in init and data order (the pilot left the init unseeded);
  * a checkpoint every SAVE_STEPS and a stop on the time budget; the next session resumes when this kernel's output
    is attached (runs that finished are copied over and skipped);
  * per-example held-out losses after training, in the format of scripts/heldout_per_example.py, so Stage 6 only
    has to run the paired bootstrap. The base model's losses are computed once.

Kaggle: GPU T4 x2, Internet ON, dataset habiburrahman118/bn-full-arms (+ earlier bn-full-s05-lora-NN outputs to resume).
Outputs under /kaggle/working/bangla_filtering/full/lora/:
    arm_<X>_seed<S>/   adapter, training_report.json, heldout_per_example.csv (checkpoint-* only while unfinished)
    base_heldout_per_example.csv, stage5_summary.json
"""
import os, sys, glob, json, time, shutil, subprocess
from pathlib import Path

OUT = Path("/kaggle/working/bangla_filtering/full/lora")
TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", "11"))
FINISH_MARGIN_MIN = 20          # held-out eval + generations + saving after the training loop stops
MIN_START_MIN = 30              # do not start a new run with less time than this left

# Edit per session. GPU 0 and GPU 1 each run their queue in order. Tokens (Stage 4 v4): B 12.2M, C 15.5M, D 15.5M,
# E 15.2M, F 10.9M. Session 01 took B 131, C 171, D 176, E 162, F 116 min of training.
# Session 01: {0: [C42, D42], 1: [B42, E42, F42]}. Session 02: {0: [C43, B43], 1: [C44, B44]} (318 min).
# Session 03: A paused at step 11,798 / 22,780; D43, E43, F43, F44 completed.
# Session 04: resume A (~10 h) on GPU 0; finish the third seeds for D and E (~5.6 h) on GPU 1.
QUEUES = {0: [("A", 42)],
          1: [("D", 44), ("E", 44)]}
SMOKE = False                   # True: 200 rows, 20 steps, 30 held-out rows, to test the kernel in ~15 min

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 3072
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
SAVE_STEPS = 10 if SMOKE else 250       # ~15 min at the pilot rate
TEST_PROMPTS = [
    "বাংলাদেশের জাতীয় ফুলের নাম কী এবং এর বৈশিষ্ট্যগুলো কী কী?",
    "একটি সংখ্যা মৌলিক কিনা তা পরীক্ষা করার জন্য একটি পাইথন ফাংশন লেখো।",
]
N_HELDOUT_PROMPTS = 2


def find_file(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{name} not under /kaggle/input: attach the dataset habiburrahman118/bn-full-arms")
    return hits[0]


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows



# ----------------------------------------------------------------------------------------------------
# worker: python s05_lora.py worker <gpu>   (CUDA_VISIBLE_DEVICES set by the parent)
# ----------------------------------------------------------------------------------------------------
def worker(gpu, deadline):
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                              DataCollatorForSeq2Seq, TrainerCallback, set_seed)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(instruction, response):
        user = [{"role": "user", "content": instruction}]
        prompt_text = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full_text = tok.apply_chat_template(user + [{"role": "assistant", "content": response}], tokenize=False)
        assert full_text.startswith(prompt_text)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
        return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids

    def tokenize_records(records):
        rows, truncated, dropped = [], 0, 0
        for r in records:
            input_ids, labels = encode(r["instruction"], r["output"])
            if len(input_ids) > MAX_LENGTH:
                truncated += 1
                input_ids, labels = input_ids[:MAX_LENGTH], labels[:MAX_LENGTH]
            if all(l == -100 for l in labels):          # prompt fills the whole window: nothing to learn
                dropped += 1
                continue
            rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
        lengths = sorted(len(r["input_ids"]) for r in rows)
        stats = {"examples_in": len(records), "examples_used": len(rows), "truncated_to_max_length": truncated,
                 "dropped_prompt_too_long": dropped, "median_tokens": lengths[len(lengths) // 2] if lengths else 0,
                 "max_tokens": lengths[-1] if lengths else 0, "total_tokens": int(sum(lengths))}
        return Dataset.from_list(rows), stats

    @torch.no_grad()
    def example_losses(model, heldout):           # = scripts/heldout_per_example.py
        model.eval()
        out = []
        for h in heldout:
            input_ids, labels = encode(h["instruction"], h["output"])
            input_ids = torch.tensor([input_ids[:MAX_LENGTH]], device="cuda")
            labels = torch.tensor([labels[:MAX_LENGTH]], device="cuda")
            loss = model(input_ids=input_ids, labels=labels).loss.item()     # fp32, as in the pilot eval
            out.append((loss, int((labels[0, 1:] != -100).sum())))
        return out

    def write_losses(path, name, heldout, losses):
        import csv
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "id", "domain", "loss", "response_tokens"])
            for h, (loss, n) in zip(heldout, losses):
                w.writerow([name, h.get("id"), h.get("domain"), round(loss, 6), n])

    @torch.no_grad()
    def generate(model, prompt):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to("cuda")
        model.eval()
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False, use_cache=True, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def load_base():
        # .float(): fp32 master weights; new transformers would load Qwen in bf16, which a T4 only emulates
        return AutoModelForCausalLM.from_pretrained(MODEL_NAME).float().to("cuda")

    heldout = load_jsonl(find_file("heldout.jsonl"), 30 if SMOKE else None)
    prompts = TEST_PROMPTS + [h["instruction"] for h in heldout[:N_HELDOUT_PROMPTS]]

    base_csv = OUT / "base_heldout_per_example.csv"
    if gpu == 0 and not base_csv.exists():
        t = time.time()
        base = load_base()
        losses = example_losses(base, heldout)
        write_losses(base_csv, "base", heldout, losses)
        print(f"[gpu {gpu}] base held-out loss {sum(l for l, _ in losses) / len(losses):.4f} "
              f"({(time.time() - t) / 60:.1f} min)", flush=True)
        del base; torch.cuda.empty_cache()

    class TimeBudget(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if time.time() > deadline - FINISH_MARGIN_MIN * 60:
                control.should_save = True
                control.should_training_stop = True

    for arm, seed in QUEUES.get(gpu, []):
        name = f"arm_{arm}_seed{seed}"
        run_dir = OUT / name
        if (run_dir / "training_report.json").exists() and json.loads((run_dir / "training_report.json").read_text())["complete"]:
            print(f"[gpu {gpu}] {name}: already complete, skipped", flush=True)
            continue
        if time.time() > deadline - MIN_START_MIN * 60:
            print(f"[gpu {gpu}] {name}: not enough time left, not started", flush=True)
            continue
        t0 = time.time()
        records = load_jsonl(find_file(f"arm_{arm}.jsonl"), 200 if SMOKE else None)
        set_seed(seed)
        model = load_base()
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        train_ds, token_stats = tokenize_records(records)
        print(f"[gpu {gpu}] {name}: {json.dumps(token_stats)}", flush=True)

        # main() copied earlier sessions' run dirs here; the TimeBudget stop saved a full checkpoint (optimizer included)
        prev_ckpts = glob.glob(f"{run_dir}/checkpoint-*")
        resume = max(prev_ckpts, key=lambda c: int(c.rsplit("-", 1)[1])) if prev_ckpts else None
        print(f"[gpu {gpu}] {name}: resume from {resume}", flush=True)

        args = TrainingArguments(
            output_dir=str(run_dir), num_train_epochs=NUM_EPOCHS, max_steps=20 if SMOKE else -1,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=LEARNING_RATE, logging_steps=25, save_strategy="steps", save_steps=SAVE_STEPS,
            save_total_limit=1, report_to="none", fp16=True, bf16=False, dataloader_num_workers=0,
            remove_unused_columns=False, seed=seed, disable_tqdm=True)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                          data_collator=DataCollatorForSeq2Seq(tokenizer=tok, padding=True, label_pad_token_id=-100),
                          callbacks=[TimeBudget()])
        t_train = time.time()
        result = trainer.train(resume_from_checkpoint=resume)
        train_min = (time.time() - t_train) / 60
        complete = trainer.state.global_step >= trainer.state.max_steps
        report = {"model": MODEL_NAME, "arm": arm, "seed": seed, "smoke": SMOKE, "complete": complete,
                  "resumed_from": resume, "token_stats": token_stats, "max_length": MAX_LENGTH, "epochs": NUM_EPOCHS,
                  "lora": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT}, "learning_rate": LEARNING_RATE,
                  "global_steps": trainer.state.global_step, "max_steps": trainer.state.max_steps,
                  "train_loss": result.training_loss, "training_minutes_this_session": round(train_min, 2),
                  "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  "log_history": trainer.state.log_history}
        if complete:
            model.save_pretrained(run_dir)              # first, so a failing eval cannot lose the trained adapter
            for c in glob.glob(f"{run_dir}/checkpoint-*"):
                shutil.rmtree(c)
            losses = example_losses(model, heldout)
            write_losses(run_dir / "heldout_per_example.csv", name, heldout, losses)
            report["heldout_loss_after"] = sum(l for l, _ in losses) / len(losses)
            report["generations_after"] = {p: generate(model, p) for p in prompts}
        (run_dir / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[gpu {gpu}] {name}: {'DONE' if complete else 'STOPPED'} at step {trainer.state.global_step:,} / "
              f"{trainer.state.max_steps:,}, train loss {result.training_loss:.4f}, "
              f"held-out {report.get('heldout_loss_after', float('nan')):.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)
        del trainer, model; torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------------------------------
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

    # carry over what earlier sessions produced: finished runs, unfinished checkpoints, the base losses
    for d in sorted(glob.glob("/kaggle/input/**/full/lora/arm_*_seed*", recursive=True)):
        dst = OUT / Path(d).name
        if not dst.exists():
            shutil.copytree(d, dst)
            print("carried over", d, flush=True)
    for f in glob.glob("/kaggle/input/**/full/lora/base_heldout_per_example.csv", recursive=True)[:1]:
        shutil.copy(f, OUT / "base_heldout_per_example.csv")

    import torch
    n_gpu = torch.cuda.device_count()
    print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], "| queues:", QUEUES, flush=True)
    if n_gpu < len(QUEUES):
        print(f"WARNING: {n_gpu} GPU(s) for {len(QUEUES)} queues; queues without a GPU do not run", flush=True)
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "DEADLINE": str(deadline)}
    workers = [subprocess.Popen([sys.executable, __file__, "worker", str(g)], env={**env, "CUDA_VISIBLE_DEVICES": str(g)})
               for g in sorted(QUEUES) if g < n_gpu]
    print("workers exited with", [w.wait() for w in workers], flush=True)

    runs = {}
    for p in sorted(OUT.glob("arm_*_seed*/training_report.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs[p.parent.name] = {k: r.get(k) for k in ("complete", "global_steps", "max_steps", "train_loss",
                                                     "heldout_loss_after", "training_minutes_this_session", "peak_gpu_gib")}
    summary = {"runs": runs, "queued": {str(g): [f"arm_{a}_seed{s}" for a, s in q] for g, q in QUEUES.items()},
               "smoke": SMOKE, "total_minutes": round((time.time() - T0) / 60, 1)}
    (OUT / "stage5_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    todo = [f"arm_{a}_seed{s}" for q in QUEUES.values() for a, s in q
            if not runs.get(f"arm_{a}_seed{s}", {}).get("complete")]
    if todo:
        print("INCOMPLETE:", todo, "-> push the next bn-full-s05-lora-NN with this output attached")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), float(os.environ["DEADLINE"]))
    else:
        main()
