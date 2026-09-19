from datasets import load_dataset

DATASET_NAME = "md-nishat-008/Bangla-Instruct"

print("Loading dataset...")

dataset = load_dataset(DATASET_NAME)

print("\nDataset:")
print(dataset)

print("\nSplits:")
print(dataset.keys())

train_data = dataset["train"]

print("\nNumber of rows:")
print(len(train_data))

print("\nColumns:")
print(train_data.column_names)

print("\nDataset summary")
print("================")

print("Rows:", len(train_data))
print("Columns:", train_data.column_names)

for i in range(5):
    instruction = train_data[i]["instruction"]
    response = train_data[i]["response"]

    print(f"\nExample {i}")
    print("Instruction length:", len(instruction))
    print("Response length:", len(response))