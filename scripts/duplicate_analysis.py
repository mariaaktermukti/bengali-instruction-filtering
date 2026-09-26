import json
import re
import unicodedata
from difflib import SequenceMatcher

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/pilot/original_tigerllm_pilot_1000.jsonl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SAMPLE_SIZE = 900


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def lexical_similarity(text1, text2):
    return SequenceMatcher(
        None,
        normalize_text(text1),
        normalize_text(text2)
    ).ratio()


# --------------------------------------------------
# Load data
# --------------------------------------------------

data = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

data = data[:SAMPLE_SIZE]

instructions = []

for item in data:
    instructions.append(item["instruction"])


print("Total samples:", len(data))


# --------------------------------------------------
# Exact duplicate analysis
# --------------------------------------------------

normalized_instructions = []

for instruction in instructions:
    normalized_instructions.append(
        normalize_text(instruction)
    )


seen = {}
duplicate_groups = []

for i in range(len(normalized_instructions)):

    text = normalized_instructions[i]

    if text in seen:
        duplicate_groups.append(
            (seen[text], i)
        )
    else:
        seen[text] = i


print()
print("========== EXACT DUPLICATES ==========")
print("Duplicate pairs:", len(duplicate_groups))


for pair in duplicate_groups[:10]:

    i = pair[0]
    j = pair[1]

    print()
    print("Index", i, "and", j)
    print("Instruction:", instructions[i])


# --------------------------------------------------
# Generate instruction embeddings
# --------------------------------------------------

print()
print("Generating instruction embeddings...")

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    instructions,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# Pairwise similarity
# --------------------------------------------------

similar_pairs = []

for i in range(len(data)):

    for j in range(i + 1, len(data)):

        semantic_similarity = float(
            np.dot(
                embeddings[i],
                embeddings[j]
            )
        )

        lexical = lexical_similarity(
            instructions[i],
            instructions[j]
        )

        similar_pairs.append(
            (
                semantic_similarity,
                lexical,
                i,
                j
            )
        )


# Sort by semantic similarity

similar_pairs.sort(
    key=lambda x: x[0],
    reverse=True
)


# --------------------------------------------------
# Top semantic pairs
# --------------------------------------------------

print()
print("========== TOP SEMANTICALLY SIMILAR PAIRS ==========")

for rank, pair in enumerate(similar_pairs[:20], start=1):

    semantic = pair[0]
    lexical = pair[1]
    i = pair[2]
    j = pair[3]

    print()
    print("Rank:", rank)
    print("Semantic similarity:", round(semantic, 4))
    print("Lexical similarity:", round(lexical, 4))

    print("A:", instructions[i])
    print("B:", instructions[j])


# --------------------------------------------------
# Hybrid candidate detection
# --------------------------------------------------

candidates = []

for pair in similar_pairs:

    semantic = pair[0]
    lexical = pair[1]
    i = pair[2]
    j = pair[3]

    # Exploratory candidate-generation rule.
    # NOT a final filtering threshold.

    if semantic >= 0.90 and lexical >= 0.60:

        candidates.append(pair)


print()
print("========== HYBRID NEAR-DUPLICATE CANDIDATES ==========")
print(
    "Semantic >= 0.90 AND lexical >= 0.60:",
    len(candidates)
)


for rank, pair in enumerate(candidates[:30], start=1):

    semantic = pair[0]
    lexical = pair[1]
    i = pair[2]
    j = pair[3]

    print()
    print("Candidate:", rank)
    print("Semantic similarity:", round(semantic, 4))
    print("Lexical similarity:", round(lexical, 4))

    print("A:", instructions[i])
    print("B:", instructions[j])


print()
print("Duplicate analysis completed.")