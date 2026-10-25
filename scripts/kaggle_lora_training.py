"""Kaggle-ready version of scripts/toy_lora_training.py, training on one arm.

Run it on Kaggle (GPU T4 + Internet ON) with the output of the
`bangla-filtering-selection` notebook attached as input. kaggle/run_arms.sh
pushes one script kernel per arm, setting ARM below; by hand:

    !python kaggle_lora_training.py

Changes vs the original script:
  * trains on arm_C.jsonl (DEITA-style selection) and evaluates on heldout.jsonl,
    both found anywhere under /kaggle/input (or via DATA_FILE / HELDOUT_FILE)
  * removes Kaggle's old torchao (0.10), which makes peft's get_peft_model crash
  * pins training to one GPU (Kaggle "T4 x2" otherwise triggers DataParallel)
  * fp16 mixed precision on GPU (T4 has no native bf16), fp32 master weights
  * MAX_LENGTH 3072 + gradient checkpointing: Qwen needs ~1,160 tokens for the
    median arm C example, so 512 truncated 351 of 400 rows
  * real Qwen chat template + loss on the response only (prompt tokens masked)
  * held-out loss before/after training, before/after generations, JSON report
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import glob
import json
import math
import subprocess
import sys
import time

from importlib.metadata import PackageNotFoundError, version

# Kaggle ships torchao 0.10.0; recent peft refuses to build LoRA layers when it
# sees torchao < 0.16 ("Found an incompatible version of torchao"). LoRA does not
# need torchao, so remove the old one before peft is imported.
try:
    torchao_version = version("torchao")
except PackageNotFoundError:
    torchao_version = None

if torchao_version and tuple(int(x) for x in torchao_version.split(".")[:2]) < (0, 16):
    print("Removing incompatible torchao", torchao_version)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])

try:
    import peft  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "peft"])

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ARM = "C"
DATA_FILE_NAME = f"arm_{ARM}.jsonl"
HELDOUT_FILE_NAME = "heldout.jsonl"

ON_KAGGLE = os.path.exists("/kaggle/working")

OUTPUT_DIR = (
    f"/kaggle/working/outputs/toy_lora_arm_{ARM}"
    if ON_KAGGLE
    else f"outputs/toy_lora_arm_{ARM}"
)

SEED = 42

MAX_LENGTH = 3072
NUM_EPOCHS = 1
MAX_STEPS = int(os.environ.get("MAX_STEPS", "-1"))  # -1 = use NUM_EPOCHS

LEARNING_RATE = 1e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

USE_FP16 = torch.cuda.is_available()  # set False if the loss turns into nan

TEST_PROMPTS = [
    "বাংলাদেশের জাতীয় ফুলের নাম কী এবং এর বৈশিষ্ট্যগুলো কী কী?",
    "একটি সংখ্যা মৌলিক কিনা তা পরীক্ষা করার জন্য একটি পাইথন ফাংশন লেখো।",
]
N_HELDOUT_PROMPTS = 2  # also generate for the first N held-out instructions


def find_file(file_name, env_var):

    if os.environ.get(env_var):
        return os.environ[env_var]

    candidates = sorted(
        glob.glob(f"/kaggle/input/**/{file_name}", recursive=True)
        + glob.glob(f"data/**/{file_name}", recursive=True)
    )

    if not candidates:
        raise FileNotFoundError(
            f"{file_name} not found under /kaggle/input. Attach the output of the "
            f"bangla-filtering-selection notebook (Add Input -> Your Work) "
            f"or set {env_var}."
        )

    if len(candidates) > 1:
        print(f"Several {file_name} found, using the first:", candidates)

    return candidates[0]


def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            records.append({
                "instruction": item["instruction"],
                "response": item.get("response", item.get("output"))
            })

    return records


def tokenize_records(records, tokenizer):

    rows = []
    truncated = 0
    dropped = 0

    for item in records:

        user = [{"role": "user", "content": item["instruction"]}]

        prompt_text = tokenizer.apply_chat_template(
            user,
            tokenize=False,
            add_generation_prompt=True
        )

        full_text = tokenizer.apply_chat_template(
            user + [{"role": "assistant", "content": item["response"]}],
            tokenize=False
        )

        assert full_text.startswith(prompt_text)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(
            full_text[len(prompt_text):],
            add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        if len(input_ids) > MAX_LENGTH:
            truncated += 1
            input_ids = input_ids[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]

        # Prompt fills the whole window: nothing left to learn from
        if all(label == -100 for label in labels):
            dropped += 1
            continue

        rows.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        })

    lengths = sorted(len(r["input_ids"]) for r in rows)

    stats = {
        "examples_in": len(records),
        "examples_used": len(rows),
        "truncated_to_max_length": truncated,
        "dropped_prompt_too_long": dropped,
        "median_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "max_tokens": lengths[-1] if lengths else 0
    }

    return Dataset.from_list(rows), stats


@torch.no_grad()
def generate(model, tokenizer, prompt):

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    model.eval()

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id
    )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )


def main():

    print("=" * 60)
    print(f"Toy LoRA Feasibility Test (Kaggle) - arm {ARM}")
    print("=" * 60)

    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no GPU. Set Accelerator = GPU T4 in notebook settings.")

    print("\nLoading dataset...")

    data_file = find_file(DATA_FILE_NAME, "DATA_FILE")
    heldout_file = find_file(HELDOUT_FILE_NAME, "HELDOUT_FILE")

    print("Train file:  ", data_file)
    print("Held-out file:", heldout_file)

    records = load_jsonl(data_file)
    heldout = load_jsonl(heldout_file)

    print("Train examples:", len(records))
    print("Held-out examples:", len(heldout))

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    # .float() instead of a dtype argument: works on old and new transformers,
    # where the default dtype changed to "auto" (bf16 for Qwen, slow on T4)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).float()

    if torch.cuda.is_available():
        model = model.to("cuda")

    model.config.use_cache = False

    print("\nApplying LoRA...")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    print("\nTokenizing dataset...")

    tokenized_dataset, token_stats = tokenize_records(records, tokenizer)
    tokenized_heldout, heldout_stats = tokenize_records(heldout, tokenizer)

    print("Train:", json.dumps(token_stats, indent=2))
    print("Held-out:", json.dumps(heldout_stats, indent=2))

    prompts = TEST_PROMPTS + [h["instruction"] for h in heldout[:N_HELDOUT_PROMPTS]]

    print("\nGenerations BEFORE training (adapter disabled)...")

    before = {}

    with model.disable_adapter():
        for prompt in prompts:
            before[prompt] = generate(model, tokenizer, prompt)
            print("\nQ:", prompt[:300], "\nA:", before[prompt])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        max_steps=MAX_STEPS,

        per_device_train_batch_size=1,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=4,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={"use_reentrant": False},

        learning_rate=LEARNING_RATE,

        logging_steps=10,

        save_strategy="no",

        report_to="none",

        fp16=USE_FP16,

        bf16=False,

        dataloader_num_workers=0,

        remove_unused_columns=False,

        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_heldout,
        data_collator=data_collator
    )

    print("\nHeld-out loss BEFORE training...")

    # LoRA B matrices start at zero, so this is the base model's loss
    eval_before = trainer.evaluate()["eval_loss"]

    print("Held-out loss:", round(eval_before, 4), "| ppl:", round(math.exp(eval_before), 2))

    print("\nStarting training...")

    model.train()

    start_time = time.time()

    train_result = trainer.train()

    training_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training completed")
    print("=" * 60)

    print(
        "Training time:",
        round(training_time / 60, 2),
        "minutes"
    )

    print("Train loss:", round(train_result.training_loss, 4))

    eval_after = trainer.evaluate()["eval_loss"]

    print(
        "Held-out loss:",
        round(eval_before, 4), "->", round(eval_after, 4),
        "| ppl:",
        round(math.exp(eval_before), 2), "->", round(math.exp(eval_after), 2)
    )

    if torch.cuda.is_available():
        print("Peak GPU memory:", round(torch.cuda.max_memory_allocated() / 2**30, 2), "GiB")

    print("\nGenerations AFTER training...")

    after = {}

    for prompt in prompts:
        after[prompt] = generate(model, tokenizer, prompt)
        print("\nQ:", prompt[:300], "\nA:", after[prompt])

    print("\nSaving LoRA adapter...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    report = {
        "model": MODEL_NAME,
        "arm": ARM,
        "train_file": data_file,
        "heldout_file": heldout_file,
        "token_stats": token_stats,
        "heldout_token_stats": heldout_stats,
        "max_length": MAX_LENGTH,
        "epochs": NUM_EPOCHS,
        "max_steps": MAX_STEPS,
        "global_steps": train_result.global_step,
        "train_loss": train_result.training_loss,
        "heldout_loss_before": eval_before,
        "heldout_loss_after": eval_after,
        "log_history": trainer.state.log_history,
        "training_minutes": round(training_time / 60, 2),
        "fp16": USE_FP16,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_gpu_gib": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if torch.cuda.is_available()
            else None
        ),
        "generations_before": before,
        "generations_after": after
    }

    with open(os.path.join(OUTPUT_DIR, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
