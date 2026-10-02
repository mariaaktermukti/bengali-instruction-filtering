import json
import numpy as np


METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"


def load_data():
    data = []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


def word_count(text):
    return len(text.split())


def unique_word_ratio(text):
    words = text.split()

    if len(words) == 0:
        return 0.0

    unique_words = set(words)

    return len(unique_words) / len(words)


def main():

    print("Loading metadata...")

    data = load_data()

    print("Total samples:", len(data))

    instruction_lengths = []
    response_lengths = []
    ratios = []
    diversity_scores = []

    for item in data:

        instruction = item["instruction"]
        response = item["response"]

        instruction_length = word_count(instruction)
        response_length = word_count(response)

        instruction_lengths.append(instruction_length)
        response_lengths.append(response_length)

        if instruction_length > 0:
            ratios.append(response_length / instruction_length)

        diversity_scores.append(
            unique_word_ratio(response)
        )

    instruction_lengths = np.array(instruction_lengths)
    response_lengths = np.array(response_lengths)
    ratios = np.array(ratios)
    diversity_scores = np.array(diversity_scores)

    print()
    print("Instruction Length")
    print("--------------------------------")

    print("Minimum:", np.min(instruction_lengths))
    print("Maximum:", np.max(instruction_lengths))
    print("Mean:", round(float(np.mean(instruction_lengths)), 2))
    print("Median:", round(float(np.median(instruction_lengths)), 2))

    print()
    print("Response Length")
    print("--------------------------------")

    print("Minimum:", np.min(response_lengths))
    print("Maximum:", np.max(response_lengths))
    print("Mean:", round(float(np.mean(response_lengths)), 2))
    print("Median:", round(float(np.median(response_lengths)), 2))

    print()
    print("Response / Instruction Ratio")
    print("--------------------------------")

    print("Minimum:", round(float(np.min(ratios)), 2))
    print("Maximum:", round(float(np.max(ratios)), 2))
    print("Mean:", round(float(np.mean(ratios)), 2))
    print("Median:", round(float(np.median(ratios)), 2))

    print()
    print("Response Lexical Diversity")
    print("--------------------------------")

    print("Minimum:", round(float(np.min(diversity_scores)), 4))
    print("Maximum:", round(float(np.max(diversity_scores)), 4))
    print("Mean:", round(float(np.mean(diversity_scores)), 4))
    print("Median:", round(float(np.median(diversity_scores)), 4))

    print()
    print("Quality feature analysis completed.")


if __name__ == "__main__":
    main()