import json
from collections import defaultdict

INPUT_PATH = "data/pilot/original_tigerllm_pilot_1000.jsonl"

examples = []

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        examples.append(json.loads(line))

instruction_groups = defaultdict(list)

for index, example in enumerate(examples):
    instruction = example["instruction"].strip()
    instruction_groups[instruction].append(index)

duplicates = {
    instruction: indices
    for instruction, indices in instruction_groups.items()
    if len(indices) > 1
}

print("Duplicate Instruction Inspection")
print("================================")

print("Duplicate instruction groups:", len(duplicates))

for group_number, (instruction, indices) in enumerate(
    duplicates.items(),
    start=1
):
    print(f"\n--- Group {group_number} ---")

    print("Instruction:")
    print(instruction)

    for index in indices:
        print(f"\nExample index: {index}")
        print("Response:")
        print(examples[index]["response"])

    print("Number of occurrences:", len(indices))