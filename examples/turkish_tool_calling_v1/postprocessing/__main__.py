import json
from .streamer import get_dirs, get_jsonl_files, read_lines, save_line
from pathlib import Path
from typing import Optional


def postprocess(line: str, max_newlines: int = 10) -> Optional[dict]:
    """Postprocess the given line string by limiting consecutive newlines."""
    try:
        if line.count(r"\n") > max_newlines:
            raise ValueError("Too many newlines in the line.")
        return json.loads(line)
    except Exception:
        return None


def main():
    success = 0
    failed = 0
    base_dir = Path.cwd()
    output = base_dir / "postprocessed.jsonl"
    output.touch(exist_ok=True)
    dirs = get_dirs(base_dir)
    for dir in dirs:
        jsonl_files = get_jsonl_files(dir)
        for jsonl_file in jsonl_files:
            for line in read_lines(jsonl_file):
                json_dict = postprocess(line)
                if json_dict is not None:
                    success += 1
                    json_dict["id"] = f"record_{success:06d}"
                    save_line(output, json.dumps(json_dict, ensure_ascii=False))
                else:
                    failed += 1
                print(
                    f"\rProcessed lines: {success + failed} (Success: {success}, Failed: {failed})",
                    end="\r",
                )
    print(f"\nTotal processed lines: {success}")
    print(f"Total failed lines: {failed}")


if __name__ == "__main__":
    main()
