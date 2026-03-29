import subprocess
import sys
from pathlib import Path


def test_cli_import_does_not_eagerly_load_optional_modules():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-W", "default", "-c", "import book_maker.cli"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "google.generativeai" not in result.stderr
    assert "gemini_translator.py" not in result.stderr
    assert "invalid escape sequence" not in result.stderr
    assert "srt_loader.py" not in result.stderr