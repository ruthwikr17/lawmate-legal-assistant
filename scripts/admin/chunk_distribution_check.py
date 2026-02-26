# Count chunk distribution by document_type
# Used for validation & debugging


import json
from collections import Counter

counter = Counter()

with open("storage/processed_chunks/chunks.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        counter[data["metadata"]["document_type"]] += 1

print(counter)
