import json
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DATA_FILE = "scoring/results/llm_judge_900.jsonl"
OUTPUT_EMBEDDINGS = "data/pilot/tigerllm_embeddings_900.npy"
OUTPUT_METADATA = "data/pilot/tigerllm_metadata_900.jsonl"


def load_data():
    texts = []
    metadata = []

    with open(DATA_FILE, "r", encoding="utf-8") as file:

        for index, line in enumerate(file):

            item = json.loads(line)

            instruction = item["instruction"]
            response = item["response"]

            text = (
                "Instruction: " + instruction +
                "\nResponse: " + response
            )

            texts.append(text)

            metadata.append({
                "index": index,
                "instruction": instruction,
                "response": response,
                "llm_complexity": item["llm_complexity"],
                "llm_quality": item["llm_quality"],
                "reason": item["reason"]
            })

    return texts, metadata


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading LLM-judged 900 data...")

    texts, metadata = load_data()

    print("Examples loaded:", len(texts))

    print()
    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print()
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])

    print()
    print("Saving embeddings...")

    np.save(OUTPUT_EMBEDDINGS, embeddings)

    print("Saved:", OUTPUT_EMBEDDINGS)

    print()
    print("Saving metadata...")

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8"
    ) as file:

        for item in metadata:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    print("Saved:", OUTPUT_METADATA)

    print()
    print("Embedding generation completed successfully.")


if __name__ == "__main__":
    main()