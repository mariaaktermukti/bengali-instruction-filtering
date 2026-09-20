import json


INPUT_FILE = "data/pilot/original_tigerllm_pilot_1000.jsonl"
VALIDATION_FILE = "data/pilot/judge_validation_100.jsonl"
OUTPUT_FILE = "data/pilot/pilot_remaining_900.jsonl"


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


# Load files
pilot = load_jsonl(INPUT_FILE)
validation = load_jsonl(VALIDATION_FILE)


# Create unique keys using instruction + response
validation_keys = set()

for row in validation:
    key = (
        row["instruction"],
        row["response"]
    )

    validation_keys.add(key)


# Remove validation examples
remaining = []

for row in pilot:
    key = (
        row["instruction"],
        row["response"]
    )

    if key not in validation_keys:
        remaining.append(row)


# Save remaining 900
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for row in remaining:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


print("=" * 60)
print("REMAINING PILOT DATA PREPARED")
print("=" * 60)

print(f"Original pilot rows:       {len(pilot)}")
print(f"Validation rows:           {len(validation)}")
print(f"Validation unique pairs:   {len(validation_keys)}")
print(f"Remaining rows:             {len(remaining)}")

print()
print(f"Saved: {OUTPUT_FILE}")
print("=" * 60)