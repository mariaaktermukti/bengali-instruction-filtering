import json
import os

INPUT_PATH = "data/pilot/judge_validation_100.jsonl"
OUTPUT_PATH = "data/pilot/human_annotation_100.jsonl"

print("Loading validation set...")

examples = []

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        examples.append(json.loads(line))

print("Total examples:", len(examples))

annotations = []

print("\nHuman Annotation Started")
print("Use scores from 1 to 5.")
print("Type 'q' at any score prompt to quit and save progress.")
print("=" * 60)

for index, example in enumerate(examples):

    print("\n" + "=" * 60)
    print(f"Example {index + 1}/{len(examples)}")
    print("=" * 60)

    print("\nInstruction:")
    print(example["instruction"])

    print("\nResponse:")
    print(example["response"])

    print("\nComplexity:")
    print("1 = Very simple")
    print("2 = Simple")
    print("3 = Moderate")
    print("4 = Complex")
    print("5 = Highly complex")

    while True:
        complexity = input("\nEnter Complexity (1-5): ").strip()

        if complexity.lower() == "q":
            break

        if complexity in ["1", "2", "3", "4", "5"]:
            break

        print("Invalid input. Please enter 1, 2, 3, 4, 5, or q.")

    if complexity.lower() == "q":
        print("\nStopping annotation...")
        break

    print("\nQuality:")
    print("1 = Incorrect / unusable")
    print("2 = Major problems")
    print("3 = Partially correct")
    print("4 = Good")
    print("5 = Excellent")

    while True:
        quality = input("\nEnter Quality (1-5): ").strip()

        if quality.lower() == "q":
            break

        if quality in ["1", "2", "3", "4", "5"]:
            break

        print("Invalid input. Please enter 1, 2, 3, 4, 5, or q.")

    if quality.lower() == "q":
        print("\nStopping annotation...")
        break

    reason = input(
        "\nReason (optional, press Enter to skip): "
    ).strip()

    record = {
        "id": example["id"],
        "instruction": example["instruction"],
        "response": example["response"],
        "human_complexity": int(complexity),
        "human_quality": int(quality),
        "reason": reason
    }

    annotations.append(record)

    print("\nSaved annotation.")
    print(
        f"Complexity = {complexity}, "
        f"Quality = {quality}"
    )

os.makedirs(
    "data/pilot",
    exist_ok=True
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    for record in annotations:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

print("\n" + "=" * 60)
print("Annotation completed/saved.")
print("Annotated examples:", len(annotations))
print("Saved to:", OUTPUT_PATH)
print("=" * 60)