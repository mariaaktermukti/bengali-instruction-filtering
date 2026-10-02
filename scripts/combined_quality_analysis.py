import json
import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_FILE = "data/pilot/original_tigerllm_pilot_1000.jsonl"

LIMIT = 900


def load_data():

    data = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:

        for index, line in enumerate(file):

            if index >= LIMIT:
                break

            item = json.loads(line)

            data.append({
                "index": index,
                "instruction": item["instruction"],
                "response": item["response"]
            })

    return data


def lexical_diversity(text):

    words = text.split()

    if len(words) == 0:
        return 0.0

    return len(set(words)) / len(words)


def main():

    data = load_data()

    print("Examples loaded:", len(data))

    model = SentenceTransformer(MODEL_NAME)

    instructions = []
    responses = []

    for item in data:

        instructions.append(item["instruction"])
        responses.append(item["response"])

    print("Creating instruction embeddings...")

    instruction_embeddings = model.encode(
        instructions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Creating response embeddings...")

    response_embeddings = model.encode(
        responses,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    alignment = np.sum(
        instruction_embeddings * response_embeddings,
        axis=1
    )

    instruction_lengths = []
    response_lengths = []
    ratios = []
    diversity = []

    for item in data:

        instruction_length = len(item["instruction"].split())
        response_length = len(item["response"].split())

        instruction_lengths.append(instruction_length)
        response_lengths.append(response_length)

        if instruction_length > 0:
            ratios.append(response_length / instruction_length)
        else:
            ratios.append(0)

        diversity.append(
            lexical_diversity(item["response"])
        )

    instruction_lengths = np.array(instruction_lengths)
    response_lengths = np.array(response_lengths)
    ratios = np.array(ratios)
    diversity = np.array(diversity)

    print()
    print("========== SUMMARY ==========")

    print()
    print("Instruction length")
    print("Mean:", round(float(np.mean(instruction_lengths)), 2))
    print("Median:", round(float(np.median(instruction_lengths)), 2))

    print()
    print("Response length")
    print("Mean:", round(float(np.mean(response_lengths)), 2))
    print("Median:", round(float(np.median(response_lengths)), 2))

    print()
    print("Response / Instruction ratio")
    print("Mean:", round(float(np.mean(ratios)), 2))
    print("Median:", round(float(np.median(ratios)), 2))

    print()
    print("Lexical diversity")
    print("Mean:", round(float(np.mean(diversity)), 4))
    print("Median:", round(float(np.median(diversity)), 4))

    print()
    print("Alignment")
    print("Mean:", round(float(np.mean(alignment)), 4))
    print("Median:", round(float(np.median(alignment)), 4))

    print()
    print("========== CORRELATIONS ==========")

    print(
        "Instruction length vs alignment:",
        round(float(np.corrcoef(instruction_lengths, alignment)[0, 1]), 4)
    )

    print(
        "Response length vs alignment:",
        round(float(np.corrcoef(response_lengths, alignment)[0, 1]), 4)
    )

    print(
        "Ratio vs alignment:",
        round(float(np.corrcoef(ratios, alignment)[0, 1]), 4)
    )

    print(
        "Diversity vs alignment:",
        round(float(np.corrcoef(diversity, alignment)[0, 1]), 4)
    )

    print()
    print("========== MULTI-SIGNAL OUTLIERS ==========")

    for i in range(len(data)):

        signals = 0

        if instruction_lengths[i] <= 4:
            signals += 1

        if ratios[i] >= 50:
            signals += 1

        if diversity[i] <= 0.45:
            signals += 1

        if alignment[i] <= 0.30:
            signals += 1

        if signals >= 2:

            print()
            print("Index:", data[i]["index"])
            print("Signals:", signals)
            print("Instruction length:", instruction_lengths[i])
            print("Response length:", response_lengths[i])
            print("Ratio:", round(float(ratios[i]), 2))
            print("Diversity:", round(float(diversity[i]), 4))
            print("Alignment:", round(float(alignment[i]), 4))
            print("Instruction:", data[i]["instruction"])
            print("Response:", data[i]["response"][:200])


if __name__ == "__main__":
    main()