import json
import numpy as np


EMBEDDINGS_FILE = "data/pilot/tigerllm_embeddings_900.npy"
METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"

TOP_K = 20


def load_metadata():
    metadata = []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            metadata.append(json.loads(line))

    return metadata


def main():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Loading metadata...")

    metadata = load_metadata()

    print("Calculating similarity matrix...")

    similarity_matrix = np.matmul(embeddings, embeddings.T)

    pairs = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = similarity_matrix[i][j]

            pairs.append((similarity, i, j))

    pairs.sort(reverse=True)

    print()
    print("Top", TOP_K, "Most Similar Pairs")
    print("--------------------------------")

    for rank in range(TOP_K):

        similarity, i, j = pairs[rank]

        print()
        print("Rank:", rank + 1)
        print("Similarity:", round(float(similarity), 4))
        print("Pair:", i, "<->", j)

        print()
        print("Instruction 1:")
        print(metadata[i]["instruction"])

        print()
        print("Instruction 2:")
        print(metadata[j]["instruction"])

        print()
        print("--------------------------------")


if __name__ == "__main__":
    main()