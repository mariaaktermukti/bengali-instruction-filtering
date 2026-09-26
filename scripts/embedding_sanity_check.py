import json
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DATA_FILE = "data/pilot/original_tigerllm_pilot_1000.jsonl"


def load_data(limit):
    data = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            if len(data) >= limit:
                break

            item = json.loads(line)

            instruction = item["instruction"]
            response = item["response"]

            text = "Instruction: " + instruction + "\nResponse: " + response

            data.append(text)

    return data


def main():
    print("Loading model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading Bengali pilot data...")

    texts = load_data(20)

    print("Examples loaded:", len(texts))

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print()
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])

    similarity_matrix = np.matmul(embeddings, embeddings.T)

    similarities = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarities.append(similarity_matrix[i][j])

    similarities = np.array(similarities)

    print()
    print("Pairwise similarity statistics")
    print("--------------------------------")

    print("Minimum similarity:", round(float(np.min(similarities)), 4))
    print("Maximum similarity:", round(float(np.max(similarities)), 4))
    print("Mean similarity:", round(float(np.mean(similarities)), 4))
    print("Median similarity:", round(float(np.median(similarities)), 4))

    print()
    print("Embedding sanity check completed.")


if __name__ == "__main__":
    main()