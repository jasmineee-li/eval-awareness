import json
from collections import Counter

input_file = "dataset.json"

category_counter = Counter()
eval_type_counter = Counter()
eval_category_counter = Counter()

total_lines = 0

with open(input_file, "r") as f:
    samples = [x for x in json.load(f)]

for sample in samples:
    metadata = sample.get("metadata", {})
    category = metadata.get("category")
    eval_type = metadata.get("eval_type")
    eval_category = metadata.get("eval_category")
    category_counter[category] += 1
    eval_type_counter[eval_type] += 1
    eval_category_counter[eval_category] += 1

print("\nCategory counts:")
for k, v in category_counter.items():
    print(f"  {k}: {v}")

print("\nEval Type counts:")
for k, v in eval_type_counter.items():
    if k != "":
        print(f"  {k}: {v}")

print("\nEval Category counts:")
for k, v in eval_category_counter.items():
    print(f"  {k}: {v}")
