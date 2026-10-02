import json
import numpy as np


EMBEDDING_FILE = "data/pilot/tigerllm_embeddings_900.npy"
METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"

OUTPUT_FILE = "data/pilot/diversity_selected_tau010.jsonl"

TAU = 0.10
TARGET_SIZE = 164


def load_metadata():
    metadata = []

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    return metadata


def calculate_scores(metadata):

    ranked = []

    for index, item in enumerate(metadata):

        score = (
            item["llm_complexity"]
            *
            item["llm_quality"]
        )

        ranked.append({
            "index": index,
            "score": score
        })

    ranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return ranked


def diversity_selection(
    embeddings,
    ranked,
    tau,
    target_size
):

    selected = []

    for item in ranked:

        if len(selected) >= target_size:
            break

        idx = item["index"]

        if len(selected) == 0:
            selected.append(idx)
            continue

        diverse = True

        for selected_idx in selected:

            similarity = np.dot(
                embeddings[idx],
                embeddings[selected_idx]
            )

            distance = 1 - similarity

            if distance <= tau:
                diverse = False
                break

        if diverse:
            selected.append(idx)

    return selected


def main():

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDING_FILE)

    print("Loading metadata...")
    metadata = load_metadata()

    ranked = calculate_scores(metadata)

    selected = diversity_selection(
        embeddings,
        ranked,
        TAU,
        TARGET_SIZE
    )

    print("Selected examples:", len(selected))

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for idx in selected:

            item = metadata[idx]

            output = {
                "index": idx,
                "instruction": item["instruction"],
                "response": item["response"],
                "llm_complexity": item["llm_complexity"],
                "llm_quality": item["llm_quality"],
                "score": (
                    item["llm_complexity"]
                    *
                    item["llm_quality"]
                )
            }

            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()