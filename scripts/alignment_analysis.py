import json
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_FILE = "data/pilot/original_tigerllm_pilot_1000.jsonl"

LIMIT = 900


def load_data():

    instructions = []
    responses = []
    metadata = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:

        for index, line in enumerate(file):

            if index >= LIMIT:
                break

            item = json.loads(line)

            instructions.append(item["instruction"])
            responses.append(item["response"])

            metadata.append({
                "index": index,
                "instruction": item["instruction"],
                "response": item["response"]
            })

    return instructions, responses, metadata


def main():

    print("Loading model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading data...")

    instructions, responses, metadata = load_data()

    print("Examples loaded:", len(instructions))

    print("Embedding instructions...")

    instruction_embeddings = model.encode(
        instructions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding responses...")

    response_embeddings = model.encode(
        responses,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    alignment_scores = np.sum(
        instruction_embeddings * response_embeddings,
        axis=1
    )

    print()
    print("Instruction embedding shape:", instruction_embeddings.shape)
    print("Response embedding shape:", response_embeddings.shape)

    print()
    print("Instruction-Response Alignment")
    print("Minimum:", round(float(np.min(alignment_scores)), 4))
    print("Maximum:", round(float(np.max(alignment_scores)), 4))
    print("Mean:", round(float(np.mean(alignment_scores)), 4))
    print("Median:", round(float(np.median(alignment_scores)), 4))

    print()
    print("Percentiles")
    print("10th percentile:", round(float(np.percentile(alignment_scores, 10)), 4))
    print("25th percentile:", round(float(np.percentile(alignment_scores, 25)), 4))
    print("50th percentile:", round(float(np.percentile(alignment_scores, 50)), 4))
    print("75th percentile:", round(float(np.percentile(alignment_scores, 75)), 4))
    print("90th percentile:", round(float(np.percentile(alignment_scores, 90)), 4))

    print()
    print("Lowest alignment examples")

    lowest_indices = np.argsort(alignment_scores)[:10]

    for rank, index in enumerate(lowest_indices, start=1):

        print()
        print("Rank:", rank)
        print("Index:", metadata[index]["index"])
        print("Alignment:", round(float(alignment_scores[index]), 4))
        print("Instruction:", metadata[index]["instruction"])
        print("Response:", metadata[index]["response"][:200])


if __name__ == "__main__":
    main()