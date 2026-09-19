import json
import random
import os

INPUT_PATH = "data/pilot/original_tigerllm_pilot_1000.jsonl"
OUTPUT_PATH = "data/pilot/judge_validation_100.jsonl"

SAMPLE_SIZE = 100
SEED = 42

print("Loading original pilot...")

examples = []

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        examples.append(json.loads(line))

print("Pilot size:", len(examples))

random.seed(SEED)

validation_set = random.sample(
    examples,
    SAMPLE_SIZE
)

os.makedirs(
    "data/pilot",
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    for index, example in enumerate(validation_set):

        record = {
            "id": index,
            "instruction": example["instruction"],
            "response": example["response"]
        }

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

print("\nValidation set created.")
print("Size:", len(validation_set))
print("Random seed:", SEED)
print("Saved to:", OUTPUT_PATH)