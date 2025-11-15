import time
from .streamer import stream_tools_from_datasets, save_tools_to_file
from dotenv import load_dotenv
from pathlib import Path
from typing import Generator

load_dotenv()
example_dir = Path(__file__).parent.parent
file_path = example_dir / "tools.jsonl"


def stream_wrapper(stream: Generator[dict, None, None]) -> Generator[dict, None, None]:
    total = 0
    start = time.time()
    for tool in stream:
        total += 1
        yield tool
        print(f"Processed {total} tools...", end="\r")
    end = time.time()
    print(f"Finished processing {total} tool definitions in {end - start:.2f} seconds.")


def main():
    dataset_ids = [
        "argilla/Synth-APIGen-v0.1",
        "Salesforce/xlam-function-calling-60k",
        "argilla-warehouse/python-seed-tools",
    ]
    tools = stream_tools_from_datasets(dataset_ids, debug=False)
    save_tools_to_file(stream_wrapper(tools), str(file_path))


if __name__ == "__main__":
    main()
