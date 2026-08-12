import os
import json
import random

data_dir = "data"
input_file = os.path.join(data_dir, "merged_cleaned_dataset.json")

if not os.path.exists(input_file):
    print(f"❌ Error: {input_file} not found. Please run clean_and_merge.py first!")
    exit(1)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

total = len(data)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

def save_split(filename, dataset):
    path = os.path.join(data_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"  ✅ Saved {len(dataset)} items to '{path}'")

print("✂️ Splitting merged dataset...\n")
save_split("train.json", train_data)
save_split("val.json", val_data)
save_split("test.json", test_data)

print("\n🎉 Dataset split complete! (80% Train / 10% Validation / 10% Test)")
