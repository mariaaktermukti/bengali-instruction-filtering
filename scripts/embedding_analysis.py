import json
import numpy as np


EMBEDDINGS_FILE = "data/pilot/tigerllm_embeddings_900.npy"
METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"


def load_metadata():
    metadata = []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            metadata.append(json.loads(line))

    return metadata


def main():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Embedding shape:", embeddings.shape)

    print("Loading metadata...")

    metadata = load_metadata()

    print("Metadata entries:", len(metadata))

    print()
    print("Calculating similarity matrix...")

    similarity_matrix = np.matmul(embeddings, embeddings.T)

    similarities = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarities.append(similarity_matrix[i][j])

    similarities = np.array(similarities)

    print()
    print("Pairwise Similarity Statistics")
    print("--------------------------------")

    print("Number of pairs:", len(similarities))
    print("Minimum:", round(float(np.min(similarities)), 4))
    print("Maximum:", round(float(np.max(similarities)), 4))
    print("Mean:", round(float(np.mean(similarities)), 4))
    print("Median:", round(float(np.median(similarities)), 4))

    print()
    print("Percentiles")
    print("--------------------------------")

    print("10th percentile:",
          round(float(np.percentile(similarities, 10)), 4))

    print("25th percentile:",
          round(float(np.percentile(similarities, 25)), 4))

    print("50th percentile:",
          round(float(np.percentile(similarities, 50)), 4))

    print("75th percentile:",
          round(float(np.percentile(similarities, 75)), 4))

    print("90th percentile:",
          round(float(np.percentile(similarities, 90)), 4))

    print()
    print("Embedding analysis completed.")


if __name__ == "__main__":
    main()