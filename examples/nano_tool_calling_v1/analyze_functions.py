import json
from collections import Counter

with open("output/train.jsonl", "r", encoding="utf-8") as f:
    function_counts = Counter()

    for line in f:
        record = json.loads(line)
        for tool in record.get("tools", []):
            func_name = tool.get("function", {}).get("name")
            if func_name:
                function_counts[func_name] += 1

for func, count in function_counts.most_common():
    print(f"{func}: {count}")
