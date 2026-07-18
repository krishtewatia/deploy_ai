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
        "name": ["Krish", "Rahul"],
        "age": [20, 21],
        "salary": [50000, 60000]
    })


def test_upload_valid_csv(client: TestClient, tmp_path: Path, sample_df: pd.DataFrame) -> None:
    # Save df to CSV
    csv_file = tmp_path / "valid.csv"
    sample_df.to_csv(csv_file, index=False)
    
    with open(csv_file, "rb") as f:
        response = client.post("/upload", files={"file": ("valid.csv", f, "text/csv")})
        
    assert response.status_code == 200
    data = response.json()
    
    # Verify response schema
    assert "metadata" in data
    assert "preview" in data
    
    # Verify metadata fields
    metadata = data["metadata"]
    assert metadata["file_name"] == "valid.csv"
    assert metadata["rows"] == 2
    assert metadata["columns"] == 3
    assert metadata["column_names"] == ["name", "age", "salary"]
    assert metadata["numerical_columns"] == ["age", "salary"]
    assert metadata["categorical_columns"] == ["name"]
    assert metadata["datetime_columns"] == []
    assert metadata["target_column"] is None
    
    # Verify preview fields
    preview = data["preview"]
    assert "shape" in preview
    assert preview["shape"] == [2, 3]
    assert "columns" in preview
    assert "preview" in preview
    assert len(preview["preview"]) == 2


def test_upload_valid_excel(client: TestClient, tmp_path: Path, sample_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    excel_file = tmp_path / "valid.xlsx"
    excel_file.write_bytes(b"placeholder")
    
    # Mock pd.read_excel to avoid real Excel write engine requirement
    monkeypatch.setattr(pd, "read_excel", lambda path: sample_df)
    
    with open(excel_file, "rb") as f:
        response = client.post("/upload", files={"file": ("valid.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        
    assert response.status_code == 200
    data = response.json()
    
    # Verify response schema
    assert "metadata" in data
    assert "preview" in data
    
    metadata = data["metadata"]
    assert metadata["file_name"] == "valid.xlsx"
    assert metadata["rows"] == 2
    assert metadata["columns"] == 3


def test_upload_unsupported_file_type(client: TestClient, tmp_path: Path) -> None:
    txt_file = tmp_path / "invalid.txt"
    txt_file.write_text("some,text,data\n1,2,3", encoding="utf-8")
    
    with open(txt_file, "rb") as f:
        response = client.post("/upload", files={"file": ("invalid.txt", f, "text/plain")})
        
    assert response.status_code == 400
    assert "Only .csv and .xlsx files are supported." in response.json()["detail"]


def test_upload_empty_file(client: TestClient, tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    
    with open(empty_file, "rb") as f:
        response = client.post("/upload", files={"file": ("empty.csv", f, "text/csv")})
        
    assert response.status_code == 400
    assert "empty or invalid" in response.json()["detail"]


def test_upload_oversized_file(client: TestClient) -> None:
    large_payload = b"a" * (50 * 1024 * 1024 + 1)
    
    response = client.post(
        "/upload",
        files={"file": ("large.csv", large_payload, "text/csv")}
    )
    
    assert response.status_code == 413
    assert "exceeds the maximum limit of 50 MB" in response.json()["detail"]
