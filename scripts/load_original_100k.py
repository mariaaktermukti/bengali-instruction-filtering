from datasets import load_dataset

DATASET_NAME = "md-nishat-008/Bangla-Instruct"
REVISION = "3543f13"

print("Loading original TigerLLM Bangla-Instruct...")
print("Revision:", REVISION)

dataset = load_dataset(
    DATASET_NAME,
    revision=REVISION
)

print("\nDataset:")
print(dataset)

print("\nSplits:")
print(dataset.keys())

train_data = dataset["train"]

print("\nNumber of rows:")
print(len(train_data))

print("\nColumns:")
print(train_data.column_names)

print("\nFirst example lengths:")

print(
    "Instruction:",
    len(train_data[0]["instruction"])
)

print(
    "Response:",
    len(train_data[0]["response"])
)