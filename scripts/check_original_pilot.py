import json

INPUT_PATH = "data/pilot/original_tigerllm_pilot_1000.jsonl"

examples = []

with open(INPUT_PATH, "r", encoding="utf-8") as file:
    for line in file:
        examples.append(json.loads(line))

print("Original TigerLLM Pilot Quality Check")
print("====================================")

total = len(examples)

print("Total examples:", total)

# --------------------------------------------------
# Missing values
# --------------------------------------------------

missing_instruction = 0
missing_response = 0

for example in examples:
    instruction = example.get("instruction")
    response = example.get("response")

    if not instruction or not instruction.strip():
        missing_instruction += 1

    if not response or not response.strip():
        missing_response += 1

print("\nMissing values")
print("Missing instructions:", missing_instruction)
print("Missing responses:", missing_response)

# --------------------------------------------------
# Duplicate instructions
# --------------------------------------------------

instructions = []

for example in examples:
    instructions.append(
        example["instruction"].strip()
    )

unique_instructions = set(instructions)

print("\nInstruction duplicates")
print("Unique instructions:", len(unique_instructions))
print(
    "Duplicate instructions:",
    total - len(unique_instructions)
)

# --------------------------------------------------
# Duplicate instruction-response pairs
# --------------------------------------------------

pairs = set()

for example in examples:
    instruction = example["instruction"].strip()
    response = example["response"].strip()

    pairs.add(
        (instruction, response)
    )

print("\nInstruction-response duplicates")
print("Unique pairs:", len(pairs))
print(
    "Duplicate pairs:",
    total - len(pairs)
)

# --------------------------------------------------
# Length statistics
# --------------------------------------------------

instruction_lengths = []
response_lengths = []

for example in examples:
    instruction_lengths.append(
        len(example["instruction"])
    )

    response_lengths.append(
        len(example["response"])
    )

print("\nLength statistics")

print("Instruction")
print(
    "Minimum:",
    min(instruction_lengths)
)
print(
    "Maximum:",
    max(instruction_lengths)
)
print(
    "Average:",
    round(
        sum(instruction_lengths) / total,
        2
    )
)

print("\nResponse")
print(
    "Minimum:",
    min(response_lengths)
)
print(
    "Maximum:",
    max(response_lengths)
)
print(
    "Average:",
    round(
        sum(response_lengths) / total,
        2
    )
)

# --------------------------------------------------
# Very short examples
# --------------------------------------------------

short_instructions = 0
short_responses = 0

for example in examples:

    if len(example["instruction"].strip()) < 10:
        short_instructions += 1

    if len(example["response"].strip()) < 20:
        short_responses += 1

print("\nVery short examples")

print(
    "Instructions < 10 characters:",
    short_instructions
)

print(
    "Responses < 20 characters:",
    short_responses
)

print("\nQuality check completed.")