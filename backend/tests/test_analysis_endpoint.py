from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [20, 21, 22],
        "salary": [50000, 60000, 55000]
    })


def test_analyze_valid_csv(client: TestClient, tmp_path: Path, sample_df: pd.DataFrame) -> None:
    """Test standard valid CSV file analysis endpoint."""
    csv_file = tmp_path / "valid.csv"
    sample_df.to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        response = client.post("/analysis/analyze", files={"file": ("valid.csv", f, "text/csv")})

    assert response.status_code == 200
    data = response.json()

    # Verify schema compliance
    assert "missing_values" in data
    assert "duplicates" in data
    assert "statistics" in data
    assert "imbalance" in data

    # Verify content correctness
    assert data["missing_values"]["total_missing"] == 0
    assert data["duplicates"]["duplicate_rows"] == 0
    assert "age" in data["statistics"]["numerical_summary"]
    assert data["imbalance"] is None


def test_analyze_valid_excel(client: TestClient, tmp_path: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test standard valid Excel (.xlsx) file analysis endpoint."""
    excel_file = tmp_path / "valid.xlsx"
    excel_file.write_bytes(b"placeholder")

    # Mock pd.read_excel to bypass dependency on an Excel engine package
    monkeypatch.setattr(pd, "read_excel", lambda path: sample_df)

    with open(excel_file, "rb") as f:
        response = client.post(
            "/analysis/analyze",
            files={"file": ("valid.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        )

    assert response.status_code == 200
    data = response.json()
    assert "statistics" in data
    assert "age" in data["statistics"]["numerical_summary"]


def test_analyze_unsupported_file_type(client: TestClient, tmp_path: Path) -> None:
    """Test analysis returns 400 Bad Request when an unsupported suffix is uploaded."""
    txt_file = tmp_path / "invalid.txt"
    txt_file.write_text("some,text,data", encoding="utf-8")

    with open(txt_file, "rb") as f:
        response = client.post("/analysis/analyze", files={"file": ("invalid.txt", f, "text/plain")})

    assert response.status_code == 400
    assert "Only .csv, .xlsx, and .xls files are supported" in response.json()["detail"]


def test_analyze_empty_file(client: TestClient, tmp_path: Path) -> None:
    """Test analysis returns 400 Bad Request when an empty file is processed by loaders."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with open(empty_file, "rb") as f:
        response = client.post("/analysis/analyze", files={"file": ("empty.csv", f, "text/csv")})

    assert response.status_code == 400
    assert "empty or invalid" in response.json()["detail"]


def test_analyze_oversized_file(client: TestClient) -> None:
    """Test analysis returns 413 Payload Too Large when file exceeds size constraint."""
    large_payload = b"a" * (50 * 1024 * 1024 + 1)

    response = client.post(
        "/analysis/analyze",
        files={"file": ("large.csv", large_payload, "text/csv")}
    )

    assert response.status_code == 413
    assert "exceeds the maximum limit of 50 MB" in response.json()["detail"]
