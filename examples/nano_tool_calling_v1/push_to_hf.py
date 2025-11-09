import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import DatasetCard
from dotenv import load_dotenv


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def push_to_hub(
    dataset_path: Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    readme_path: Path | None = None,
):
    train = load_jsonl(dataset_path / "train.jsonl")
    val_path = dataset_path / "val.jsonl"

    # Convert to JSON strings to avoid schema issues
    for record in train:
        record["tools"] = json.dumps(record["tools"])
        record["messages"] = json.dumps(record["messages"])
        record["assistant_calls"] = json.dumps(record["assistant_calls"])
        record["problem_metadata"] = json.dumps(record["problem_metadata"])
        record["judge"] = json.dumps(record["judge"])
        record["quality_tags"] = json.dumps(record["quality_tags"])
        record["tools_metadata"] = json.dumps(record["tools_metadata"])

    dataset = Dataset.from_list(train)

    if val_path.exists():
        val = load_jsonl(val_path)
        for record in val:
            record["tools"] = json.dumps(record["tools"])
            record["messages"] = json.dumps(record["messages"])
            record["assistant_calls"] = json.dumps(record["assistant_calls"])
            record["problem_metadata"] = json.dumps(record["problem_metadata"])
            record["judge"] = json.dumps(record["judge"])
            record["quality_tags"] = json.dumps(record["quality_tags"])
            record["tools_metadata"] = json.dumps(record["tools_metadata"])
        dataset = DatasetDict({"train": dataset, "validation": Dataset.from_list(val)})

    dataset.push_to_hub(repo_id, token=token, private=private)

    if readme_path and readme_path.exists():
        card = DatasetCard(open(readme_path, encoding="utf-8").read())
        card.push_to_hub(repo_id, token=token)

    print(f"✓ Pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    load_dotenv()
    base_path = Path(__file__).parent
    push_to_hub(
        base_path / "output",
        "atasoglu/nano-tool-calling-v1",
        os.getenv("HF_TOKEN"),
        readme_path=base_path / "README.md",
    )
