from datasets import load_dataset
import json
import os

DATASET_NAME = "md-nishat-008/Bangla-Instruct"
PILOT_SIZE = 1000
SEED = 42

OUTPUT_PATH = "data/pilot/bangla_instruct_pilot_1000.jsonl"

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME)
train_data = dataset["train"]

print("Total examples:", len(train_data))

print("\nCreating random pilot sample...")
pilot = train_data.shuffle(seed=SEED).select(range(PILOT_SIZE))

os.makedirs("data/pilot", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for example in pilot:
        record = {
            "instruction": example["instruction"],
            "response": example["response"]
        }

        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Pilot size:", len(pilot))
print("Saved to:", OUTPUT_PATH)
print("Random seed:", SEED)