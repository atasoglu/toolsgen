from pathlib import Path
import json
import subprocess
import sys


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "toolsgen.cli", *args]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_version_runs() -> None:
    result = run_cli(["version"])
    assert result.returncode == 0
    assert result.stdout.strip()  # non-empty version


def test_generate_creates_manifest(tmp_path: Path) -> None:
    tools_path = tmp_path / "tools.json"
    tools_path.write_text(json.dumps([]), encoding="utf-8")
    out_dir = tmp_path / "out"

    result = run_cli(
        [
            "generate",
            "--tools",
            str(tools_path),
            "--out",
            str(out_dir),
            "--num",
            "3",
            "--strategy",
            "random",
            "--seed",
            "42",
        ]
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_requested"] == 3
    assert manifest["strategy"] == "random"
    assert manifest["seed"] == 42
