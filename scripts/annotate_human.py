import json
import os

INPUT_PATH = "data/pilot/judge_validation_100.jsonl"
OUTPUT_PATH = "data/pilot/human_annotation_100.jsonl"


def load_jsonl(path):
    records = []

    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


print("Loading validation set...")

examples = load_jsonl(INPUT_PATH)

print("Total examples:", len(examples))


# Load previous annotations if the file already exists
annotations = load_jsonl(OUTPUT_PATH)

annotated_ids = set()

for record in annotations:
    annotated_ids.add(record["id"])


print("Already annotated:", len(annotations))
print("Remaining:", len(examples) - len(annotated_ids))


print("\nHuman Annotation Started")
print("Use scores from 1 to 5.")
print("Type 'q' at any score prompt to quit.")
print("Progress is saved after every annotation.")
print("=" * 60)


for index, example in enumerate(examples):

    example_id = example["id"]

    # Skip already annotated examples
    if example_id in annotated_ids:
        continue

    print("\n" + "=" * 60)
    print(f"Example {index + 1}/{len(examples)}")
    print(f"ID: {example_id}")
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

        complexity = input(
            "\nEnter Complexity (1-5): "
        ).strip()

        if complexity.lower() == "q":
            print("\nStopping annotation.")
            print("Your previous progress is safely saved.")
            print(
                f"Annotated examples: {len(annotations)}/{len(examples)}"
            )
            exit()

        if complexity in ["1", "2", "3", "4", "5"]:
            break

        print(
            "Invalid input. Please enter 1, 2, 3, 4, 5, or q."
        )

    print("\nQuality:")
    print("1 = Incorrect / unusable")
    print("2 = Major problems")
    print("3 = Partially correct")
    print("4 = Good")
    print("5 = Excellent")

    while True:

        quality = input(
            "\nEnter Quality (1-5): "
        ).strip()

        if quality.lower() == "q":
            print("\nStopping annotation.")
            print("Your previous progress is safely saved.")
            print(
                f"Annotated examples: {len(annotations)}/{len(examples)}"
            )
            exit()

        if quality in ["1", "2", "3", "4", "5"]:
            break

        print(
            "Invalid input. Please enter 1, 2, 3, 4, 5, or q."
        )

    reason = input(
        "\nReason (optional, press Enter to skip): "
    ).strip()

    record = {
        "id": example_id,
        "instruction": example["instruction"],
        "response": example["response"],
        "human_complexity": int(complexity),
        "human_quality": int(quality),
        "reason": reason
    }

    # Save immediately after every annotation
    with open(
        OUTPUT_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

    annotations.append(record)
    annotated_ids.add(example_id)

    print("\nAnnotation saved.")
    print(
        f"Complexity = {complexity}, "
        f"Quality = {quality}"
    )

    print(
        f"Progress: {len(annotations)}/{len(examples)}"
    )


print("\n" + "=" * 60)
print("Annotation completed.")
print(
    f"Total annotated: {len(annotations)}/{len(examples)}"
)
print("Saved to:", OUTPUT_PATH)
print("=" * 60)