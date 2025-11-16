import os
import json
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def push_to_hub(jsonl_path: Path, repo_id: str, token: str = None):
    # Load data
    data = load_jsonl(jsonl_path)

    # Process fields for HF compatibility
    for record in data:
        # Keep simple types as-is: str, int, float
        # Convert complex types (dict, list) to JSON strings
        for key, value in record.items():
            if isinstance(value, (dict, list)):
                record[key] = json.dumps(value, ensure_ascii=False)
            # Keep str, int, float, bool as-is

    # Create dataset
    dataset = Dataset.from_list(data)

    # Push to hub
    dataset.push_to_hub(repo_id, token=token)


if __name__ == "__main__":
    load_dotenv()

    base_dir = Path(__file__).parent
    jsonl_path = base_dir / "postprocessed.jsonl"
    repo_id = "atasoglu/turkish-tool-calling-10k"
    token = os.getenv("HF_TOKEN")

    push_to_hub(jsonl_path, repo_id, token)
