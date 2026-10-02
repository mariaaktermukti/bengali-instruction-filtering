import json
import time
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_FILE = "data/pilot/diversity_selected_tau010.jsonl"

OUTPUT_DIR = "outputs/toy_lora_tau010"

MAX_LENGTH = 512
NUM_EPOCHS = 1

LEARNING_RATE = 1e-4

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


def load_data():

    records = []

    with open(DATA_FILE, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            instruction = item["instruction"]
            response = item["response"]

            text = (
                "<|im_start|>user\n"
                + instruction
                + "\n<|im_end|>\n"
                + "<|im_start|>assistant\n"
                + response
                + "\n<|im_end|>"
            )

            records.append({
                "text": text
            })

    return Dataset.from_list(records)


def tokenize_dataset(dataset, tokenizer):

    def tokenize_function(examples):

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH
        )

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )


def main():

    print("=" * 60)
    print("Toy LoRA Feasibility Test")
    print("=" * 60)

    print("\nLoading dataset...")

    dataset = load_data()

    print("Number of examples:", len(dataset))

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

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

    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=4,

        learning_rate=LEARNING_RATE,

        logging_steps=10,

        save_strategy="epoch",

        report_to="none",

        fp16=False,

        bf16=False,

        dataloader_num_workers=0,

        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    print("\nStarting training...")

    start_time = time.time()

    trainer.train()

    end_time = time.time()

    training_time = end_time - start_time

    print("\n" + "=" * 60)
    print("Training completed")
    print("=" * 60)

    print(
        "Training time:",
        round(training_time / 60, 2),
        "minutes"
    )

    print("\nSaving LoRA adapter...")

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("Saved:", OUTPUT_DIR)


if __name__ == "__main__":
    main()