from pathlib import Path
from typing import Generator


def get_jsonl_files(directory: Path) -> Generator[Path, None, None]:
    """Get all JSONL files in the specified directory."""
    yield from directory.glob("*.jsonl")


def get_dirs(directory: Path) -> Generator[Path, None, None]:
    """Get all subdirectories in the specified directory."""
    for d in directory.iterdir():
        if d.is_dir():
            yield d


def read_lines(file_path: Path) -> Generator[str, None, None]:
    """Read lines from a file."""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def save_line(file_path: Path, line: str):
    """Save a line to a file."""
    with file_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_newlines(s: str) -> int:
    """Count the number of newline characters in a string."""
    return s.count("\n")
