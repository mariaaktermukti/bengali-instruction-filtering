"""Per-example held-out loss for the base model and each trained LoRA adapter.

    python scripts/heldout_per_example.py HELDOUT_JSONL OUT_CSV ARM=ADAPTER_DIR [ARM=ADAPTER_DIR ...]

On Kaggle, run with no arguments and the bengali-lora-arm-* notebooks (plus
bangla-filtering-selection) attached as inputs: the files are found automatically.

Writes one row per (model, held-out example): response-token mean cross-entropy and
the number of response tokens. The mean over examples matches the Trainer's
eval_loss (batch size 1), so arms can be compared with a paired bootstrap.
"""

import csv
import glob
import json
import os
import subprocess
import sys

from importlib.metadata import PackageNotFoundError, version

# Same Kaggle fix as kaggle_lora_training.py: peft crashes on torchao < 0.16
try:
    torchao_version = version("torchao")
except PackageNotFoundError:
    torchao_version = None

if torchao_version and tuple(int(x) for x in torchao_version.split(".")[:2]) < (0, 16):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])

import torch

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 3072

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def kaggle_arguments():

    heldout = sorted(glob.glob("/kaggle/input/**/arms/heldout.jsonl", recursive=True))[0]

    adapters = []

    for config in sorted(glob.glob("/kaggle/input/**/toy_lora_arm_*/adapter_config.json", recursive=True)):
        adapter_dir = os.path.dirname(config)
        adapters.append([adapter_dir.rsplit("toy_lora_arm_", 1)[1], adapter_dir])

    return heldout, "/kaggle/working/heldout_per_example.csv", adapters


def load_heldout(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def encode(tokenizer, instruction, response):

    user = [{"role": "user", "content": instruction}]

    prompt_text = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": response}],
        tokenize=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]

    return torch.tensor([input_ids]), torch.tensor([labels])


@torch.no_grad()
def example_losses(model, encoded):

    model.eval()

    losses = []

    for input_ids, labels in encoded:

        output = model(input_ids=input_ids.to(DEVICE), labels=labels.to(DEVICE))

        losses.append((output.loss.item(), int((labels[0, 1:] != -100).sum())))

    return losses


def main():

    if len(sys.argv) == 1 and os.path.exists("/kaggle/input"):
        heldout_path, out_path, adapters = kaggle_arguments()
    else:
        heldout_path, out_path = sys.argv[1], sys.argv[2]
        adapters = [arg.split("=", 1) for arg in sys.argv[3:]]

    print("Held-out:", heldout_path)
    print("Adapters:", adapters)

    heldout = load_heldout(heldout_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoded = [
        encode(tokenizer, h["instruction"], h.get("response", h.get("output")))
        for h in heldout
    ]

    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME).float().to(DEVICE)

    results = {"base": example_losses(base, encoded)}

    print("base", round(sum(l for l, _ in results["base"]) / len(heldout), 4))

    for arm, adapter_dir in adapters:

        model = PeftModel.from_pretrained(base, adapter_dir)

        results[arm] = example_losses(model, encoded)

        print(arm, round(sum(l for l, _ in results[arm]) / len(heldout), 4))

        base = model.unload()

    with open(out_path, "w", encoding="utf-8", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(["model", "id", "domain", "loss", "response_tokens"])

        for name, losses in results.items():
            for h, (loss, n_tokens) in zip(heldout, losses):
                writer.writerow([name, h.get("id"), h.get("domain"), round(loss, 6), n_tokens])

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
