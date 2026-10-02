import json
import numpy as np


EMBEDDING_FILE = "data/pilot/tigerllm_embeddings_900.npy"
METADATA_FILE = "data/pilot/tigerllm_metadata_900.jsonl"

TAU = 0.20
TARGET_SIZE = 164


def load_metadata():
    metadata = []

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        for line in file:
            metadata.append(json.loads(line))

    return metadata


def calculate_scores(metadata):
    scores = []

    for index, item in enumerate(metadata):

        complexity = item["llm_complexity"]
        quality = item["llm_quality"]

        score = complexity * quality

        scores.append({
            "index": index,
            "score": score
        })

    scores.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scores


def select_score_only(ranked_items, target_size):

    selected = []

    for item in ranked_items[:target_size]:
        selected.append(item["index"])

    return selected


def select_diverse(embeddings, ranked_items, tau, target_size):

    selected = []

    for item in ranked_items:

        if len(selected) >= target_size:
            break

        current_index = item["index"]

        if len(selected) == 0:
            selected.append(current_index)
            continue

        is_diverse = True

        for selected_index in selected:

            similarity = np.dot(
                embeddings[current_index],
                embeddings[selected_index]
            )

            distance = 1.0 - similarity

            if distance <= tau:
                is_diverse = False
                break

        if is_diverse:
            selected.append(current_index)

    return selected


def calculate_statistics(
    embeddings,
    metadata,
    selected
):

    selected_embeddings = embeddings[selected]

    similarity_matrix = np.matmul(
        selected_embeddings,
        selected_embeddings.T
    )

    similarities = []

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            similarities.append(
                similarity_matrix[i][j]
            )

    similarities = np.array(similarities)

    scores = []
    complexities = []
    qualities = []

    for index in selected:

        complexity = metadata[index]["llm_complexity"]
        quality = metadata[index]["llm_quality"]

        complexities.append(complexity)
        qualities.append(quality)
        scores.append(complexity * quality)

    print("Selected:", len(selected))
    print("Mean similarity:", round(float(np.mean(similarities)), 4))
    print("Max similarity:", round(float(np.max(similarities)), 4))
    print("Mean score:", round(float(np.mean(scores)), 4))
    print("Mean complexity:", round(float(np.mean(complexities)), 4))
    print("Mean quality:", round(float(np.mean(qualities)), 4))


def main():

    print("Loading embeddings...")

    embeddings = np.load(EMBEDDING_FILE)

    print("Embedding shape:", embeddings.shape)

    print()
    print("Loading metadata...")

    metadata = load_metadata()

    print("Metadata:", len(metadata))

    print()
    print("Ranking by complexity × quality...")

    ranked_items = calculate_scores(metadata)

    print()
    print("===== BASELINE: SCORE ONLY =====")

    score_only = select_score_only(
        ranked_items,
        TARGET_SIZE
    )

    calculate_statistics(
        embeddings,
        metadata,
        score_only
    )

    print()
    print("===== DIVERSITY: SCORE + DIVERSITY =====")

    diverse = select_diverse(
        embeddings,
        ranked_items,
        TAU,
        TARGET_SIZE
    )

    calculate_statistics(
        embeddings,
        metadata,
        diverse
    )

    print()
    print("Comparison completed.")


if __name__ == "__main__":
    main()