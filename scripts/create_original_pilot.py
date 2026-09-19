import json
import random
import os

INPUT_PATH = "data/raw/original_tigerllm_100k/bangla_instruct_original.jsonl"
OUTPUT_PATH = "data/pilot/original_tigerllm_pilot_1000.jsonl"

PILOT_SIZE = 1000
SEED = 42

print("Loading original TigerLLM dataset...")

examples = []

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        examples.append(json.loads(line))

print("Total examples:", len(examples))

# Reproducible random sampling
random.seed(SEED)

pilot = random.sample(
    examples,
    PILOT_SIZE
)

os.makedirs(
    "data/pilot",
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    for example in pilot:
        file.write(
            json.dumps(
                example,
                ensure_ascii=False
            ) + "\n"
        )

print("\nPilot created successfully.")
print("Pilot size:", len(pilot))
print("Random seed:", SEED)
print("Saved to:", OUTPUT_PATH)