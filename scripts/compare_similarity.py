import json
import numpy as np
from difflib import SequenceMatcher


EMBEDDINGS_FILE = "data/pilot/tigerllm_embeddings_900.npy"
METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"

TOP_K = 20


def load_metadata():
    metadata = []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            metadata.append(json.loads(line))

    return metadata


def lexical_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def main():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Loading metadata...")

    metadata = load_metadata()

    print("Calculating embedding similarity...")

    similarity_matrix = np.matmul(embeddings, embeddings.T)

    pairs = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):

            embedding_similarity = similarity_matrix[i][j]

            pairs.append(
                (embedding_similarity, i, j)
            )

    pairs.sort(reverse=True)

    print()
    print("Top", TOP_K, "Embedding-Similar Pairs")
    print("========================================")

    for rank in range(TOP_K):

        embedding_similarity, i, j = pairs[rank]

        instruction1 = metadata[i]["instruction"]
        instruction2 = metadata[j]["instruction"]

        lexical = lexical_similarity(
            instruction1,
            instruction2
        )

        print()
        print("Rank:", rank + 1)

        print(
            "Embedding similarity:",
            round(float(embedding_similarity), 4)
        )

        print(
            "Lexical similarity:",
            round(float(lexical), 4)
        )

        print("Pair:", i, "<->", j)

        print()
        print("Instruction 1:")
        print(instruction1)

        print()
        print("Instruction 2:")
        print(instruction2)

        print()
        print("----------------------------------------")


if __name__ == "__main__":
    main()