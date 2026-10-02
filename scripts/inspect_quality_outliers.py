import json


METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"


def load_data():
    data = []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


def word_count(text):
    return len(text.split())


def lexical_diversity(text):
    words = text.split()

    if len(words) == 0:
        return 0.0

    return len(set(words)) / len(words)


def main():

    data = load_data()

    samples = []

    for index, item in enumerate(data):

        instruction = item["instruction"]
        response = item["response"]

        instruction_length = word_count(instruction)
        response_length = word_count(response)

        if instruction_length > 0:
            ratio = response_length / instruction_length
        else:
            ratio = 0

        diversity = lexical_diversity(response)

        samples.append({
            "index": index,
            "instruction": instruction,
            "response": response,
            "instruction_length": instruction_length,
            "response_length": response_length,
            "ratio": ratio,
            "diversity": diversity
        })

    print()
    print("1. SHORTEST INSTRUCTIONS")
    print("================================")

    shortest = sorted(
        samples,
        key=lambda x: x["instruction_length"]
    )[:10]

    for item in shortest:

        print()
        print("Index:", item["index"])
        print("Instruction length:", item["instruction_length"])
        print("Instruction:", item["instruction"])
        print("Response:", item["response"])

    print()
    print("2. LONGEST INSTRUCTIONS")
    print("================================")

    longest = sorted(
        samples,
        key=lambda x: x["instruction_length"],
        reverse=True
    )[:10]

    for item in longest:

        print()
        print("Index:", item["index"])
        print("Instruction length:", item["instruction_length"])
        print("Instruction:", item["instruction"])
        print("Response:", item["response"])

    print()
    print("3. HIGHEST RESPONSE / INSTRUCTION RATIO")
    print("================================")

    highest_ratio = sorted(
        samples,
        key=lambda x: x["ratio"],
        reverse=True
    )[:10]

    for item in highest_ratio:

        print()
        print("Index:", item["index"])
        print("Ratio:", round(item["ratio"], 2))
        print("Instruction:", item["instruction"])
        print("Response:", item["response"])

    print()
    print("4. LOWEST RESPONSE LEXICAL DIVERSITY")
    print("================================")

    lowest_diversity = sorted(
        samples,
        key=lambda x: x["diversity"]
    )[:10]

    for item in lowest_diversity:

        print()
        print("Index:", item["index"])
        print("Lexical diversity:", round(item["diversity"], 4))
        print("Response:", item["response"])

    print()
    print("5. HIGHEST RESPONSE LEXICAL DIVERSITY")
    print("================================")

    highest_diversity = sorted(
        samples,
        key=lambda x: x["diversity"],
        reverse=True
    )[:10]

    for item in highest_diversity:

        print()
        print("Index:", item["index"])
        print("Lexical diversity:", round(item["diversity"], 4))
        print("Response:", item["response"])


if __name__ == "__main__":
    main()