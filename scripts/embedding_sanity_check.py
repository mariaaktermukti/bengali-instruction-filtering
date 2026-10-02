import numpy as np


EMBEDDING_FILE = "data/pilot/tigerllm_embeddings_900.npy"


def main():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDING_FILE)

    print("Embedding shape:", embeddings.shape)

    print()
    print("Checking normalization...")

    norms = np.linalg.norm(embeddings, axis=1)

    print("Minimum norm:", round(float(np.min(norms)), 6))
    print("Maximum norm:", round(float(np.max(norms)), 6))
    print("Mean norm:", round(float(np.mean(norms)), 6))

    print()
    print("Calculating pairwise similarities...")

    similarity_matrix = np.matmul(
        embeddings,
        embeddings.T
    )

    similarities = []

    for i in range(len(embeddings)):

        for j in range(i + 1, len(embeddings)):

            similarities.append(
                similarity_matrix[i][j]
            )

    similarities = np.array(similarities)

    print()
    print("Pairwise similarity statistics")
    print("--------------------------------")

    print(
        "Minimum similarity:",
        round(float(np.min(similarities)), 4)
    )

    print(
        "Maximum similarity:",
        round(float(np.max(similarities)), 4)
    )

    print(
        "Mean similarity:",
        round(float(np.mean(similarities)), 4)
    )

    print(
        "Median similarity:",
        round(float(np.median(similarities)), 4)
    )

    print()
    print("Percentiles")

    print(
        "25th percentile:",
        round(float(np.percentile(similarities, 25)), 4)
    )

    print(
        "75th percentile:",
        round(float(np.percentile(similarities, 75)), 4)
    )

    print(
        "95th percentile:",
        round(float(np.percentile(similarities, 95)), 4)
    )

    print()
    print("Embedding sanity check completed.")


if __name__ == "__main__":
    main()