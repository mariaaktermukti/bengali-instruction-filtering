from datasets import load_dataset
import json
import os

DATASET_NAME = "md-nishat-008/Bangla-Instruct"
REVISION = "3543f13"

OUTPUT_PATH = "data/raw/original_tigerllm_100k/bangla_instruct_original.jsonl"
PROVENANCE_PATH = "data/raw/original_tigerllm_100k/provenance.json"

print("Loading original TigerLLM Bangla-Instruct...")

dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION
)

train_data = dataset["train"]

print("Rows loaded:", len(train_data))

os.makedirs(
    "data/raw/original_tigerllm_100k",
    exist_ok=True
)

# Save dataset
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for example in train_data:
        record = {
            "instruction": example["instruction"],
            "response": example["response"]
        }

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

# Save provenance information
provenance = {
    "dataset": DATASET_NAME,
    "revision": REVISION,
    "description": "Original TigerLLM Bangla-Instruct release",
    "paper_reported_size": 100000,
    "loaded_rows": len(train_data),
    "columns": [
        "instruction",
        "response"
    ],
    "format": "JSONL"
}

with open(PROVENANCE_PATH, "w", encoding="utf-8") as file:
    json.dump(
        provenance,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\nDataset saved successfully.")
print("Dataset:", OUTPUT_PATH)
print("Provenance:", PROVENANCE_PATH)
print("Rows:", len(train_data))