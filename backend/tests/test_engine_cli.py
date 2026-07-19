"""Unit and integration tests for DeployAI engine CLI interface."""

import os
import tempfile
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add project root to sys.path to resolve root-level cli module
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli import build_arg_parser, main, run_interactive_wizard


@pytest.fixture
def sample_csv_path():
    """Create a temporary CSV file with synthetic classification data."""
    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 10,
        "feature2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        path = tmp.name

    yield path

    if os.path.exists(path):
        os.unlink(path)


def test_cli_argument_parsing(sample_csv_path):
    parser = build_arg_parser()
    args = parser.parse_args(["--dataset", sample_csv_path, "--target", "target", "--mode", "deterministic"])
    assert args.dataset == sample_csv_path
    assert args.target == "target"
    assert args.mode == "deterministic"


def test_cli_execution_success(sample_csv_path):
    os.environ["GROQ_API_KEY"] = "mock_key"
    exit_code = main(["--dataset", sample_csv_path, "--target", "target", "--mode", "deterministic"])
    assert exit_code == 0


def test_cli_execution_invalid_file():
    exit_code = main(["--dataset", "non_existent_file.csv", "--target", "target"])
    assert exit_code == 1


def test_cloud_wizard_always_prompts_for_api_key(monkeypatch):
    answers = iter([
        "1",  # datasets/iris.csv
        "5",  # target
        "Classify iris species.",
        "",   # additional context
        "",   # default pickle export
        "3",  # cloud provider
        "1",  # Groq
        "test-user-provided-key",
    ])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    _, _, mode, provider, _, _ = run_interactive_wizard()

    assert mode.value == "ai_assisted"
    assert provider.config.api_key == "test-user-provided-key"
