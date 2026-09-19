import json

INPUT_PATH = "data/pilot/bangla_instruct_pilot_1000.jsonl"

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    examples = [json.loads(line) for line in file]

print("Short Response Inspection")
print("=========================")

count = 0

for i, example in enumerate(examples):
    response = example["response"].strip()

    if len(response) < 20:
        count += 1

        print(f"\n--- Example {count} ---")
        print("Original index:", i)
        print("Instruction:")
        print(example["instruction"])

        print("\nResponse:")
        print(example["response"])

        print("\nResponse length:", len(response))

print("\nTotal short responses:", count)