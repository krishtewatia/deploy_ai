"""FastAPI integration and unit tests for the EDA Analyze endpoint.

Targets 100% coverage of backend/app/api/routes/eda.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from backend.app.analysis.analysis_service import AnalysisService
from backend.app.api.routes.eda import (
    MAX_FILE_SIZE,
    analyze_dataset,
    get_analysis_service,
    get_eda_service,
)
from backend.app.eda.eda_service import EDAService, EDAServiceError
from backend.app.eda.eda_service_schemas import EDAServiceReport
from backend.app.eda.insight_schemas import InsightReport
from backend.app.eda.schemas import (
    DatasetSummary,
    DescriptiveAnalytics,
    DiagnosticAnalytics,
)
from backend.app.main import app


# ── Fixtures & Mock Helpers ──────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    """Fixture returning a FastAPI TestClient."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_overrides():
    """Autouse fixture to clear dependency overrides before and after each test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _build_dummy_eda_report(dataset_id: str) -> EDAServiceReport:
    """Helper to build a dummy EDAServiceReport."""
    return EDAServiceReport(
        dataset_id=dataset_id,
        descriptive=DescriptiveAnalytics(
            dataset_summary=DatasetSummary(
                rows=5,
                columns=2,
                numerical_columns=["salary"],
                categorical_columns=[],
                datetime_columns=[],
                missing_cells=0,
                duplicate_rows=0,
            ),
            key_findings=[],
        ),
        diagnostic=DiagnosticAnalytics(
            correlation_findings=[],
            anomaly_findings=[],
        ),
        visualizations=[],
        generated_charts=["/mock/chart.png"],
        insights=InsightReport(
            descriptive_insights=["Line 1"],
            diagnostic_insights=[],
            predictive_observations=[],
            prescriptive_recommendations=[],
            generated_at=datetime.now(timezone.utc),
            dataset_id=dataset_id,
        ),
        generated_at=datetime.now(timezone.utc),
    )


# ── Success Tests ────────────────────────────────────────────────────────────


def test_analyze_csv_success_with_provided_id(client: TestClient, tmp_path: Path) -> None:
    """Successful CSV analysis with a custom dataset ID."""
    mock_analysis = MagicMock(spec=AnalysisService)
    mock_eda = MagicMock(spec=EDAService)

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis
    app.dependency_overrides[get_eda_service] = lambda: mock_eda

    # Set up mocks
    dummy_report = _build_dummy_eda_report("custom_id")
    mock_eda.run.return_value = dummy_report

    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"salary": [1000, 2000]}).to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        response = client.post(
            "/api/v1/eda/analyze",
            files={"file": ("data.csv", f, "text/csv")},
            data={"dataset_id": "custom_id"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"] == "custom_id"
    assert data["generated_charts"] == ["/mock/chart.png"]
    assert "insights" in data
    assert data["insights"]["descriptive_insights"] == ["Line 1"]

    mock_analysis.analyze.assert_called_once()
    mock_eda.run.assert_called_once()


def test_analyze_csv_success_generates_id(client: TestClient, tmp_path: Path) -> None:
    """Successful CSV analysis when dataset_id is not provided (should be auto-generated)."""
    mock_analysis = MagicMock(spec=AnalysisService)
    mock_eda = MagicMock(spec=EDAService)

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis
    app.dependency_overrides[get_eda_service] = lambda: mock_eda

    # Custom matcher to accept any generated string ID
    def eda_run_side_effect(dataset_id, df, analysis_report):
        assert dataset_id.startswith("test_data_")
        return _build_dummy_eda_report(dataset_id)

    mock_eda.run.side_effect = eda_run_side_effect

    csv_file = tmp_path / "test-data.csv"
    pd.DataFrame({"salary": [1000, 2000]}).to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        response = client.post(
            "/api/v1/eda/analyze",
            files={"file": ("test-data.csv", f, "text/csv")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["dataset_id"].startswith("test_data_")


# ── Error Case Tests ──────────────────────────────────────────────────────────


def test_analyze_invalid_extension(client: TestClient) -> None:
    """Uploading a non-CSV file should return 400 Bad Request."""
    file_content = b"header1,header2\n1,2\n"
    response = client.post(
        "/api/v1/eda/analyze",
        files={"file": ("data.xlsx", file_content, "application/vnd.ms-excel")},
    )
    assert response.status_code == 400
    assert "Only .csv files are supported" in response.json()["detail"]


def test_analyze_oversized_file_streaming(client: TestClient) -> None:
    """Streaming size validation should reject files > 50 MB with 413."""
    large_payload = b"a" * (50 * 1024 * 1024 + 1)
    response = client.post(
        "/api/v1/eda/analyze",
        files={"file": ("large.csv", large_payload, "text/csv")},
    )
    assert response.status_code == 413
    assert "exceeds the maximum limit of 50 MB" in response.json()["detail"]


def test_analyze_empty_file(client: TestClient, tmp_path: Path) -> None:
    """Uploading an empty file should return 400 Bad Request."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with open(empty_file, "rb") as f:
        response = client.post(
            "/api/v1/eda/analyze",
            files={"file": ("empty.csv", f, "text/csv")},
        )

    assert response.status_code == 400
    assert "empty or invalid" in response.json()["detail"]


def test_analyze_csv_parsing_failure(client: TestClient, tmp_path: Path) -> None:
    """Malformed CSV (parsing error) should return 400 Bad Request."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("header1,header2\n1,2,3,4\n5,6\n", encoding="utf-8")

    # Force pandas parser to raise a ParserError
    with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Parser boom")):
        with open(bad_csv, "rb") as f:
            response = client.post(
                "/api/v1/eda/analyze",
                files={"file": ("bad.csv", f, "text/csv")},
            )

    assert response.status_code == 400
    assert "Failed to parse CSV file" in response.json()["detail"]


def test_analyze_file_not_found(client: TestClient, tmp_path: Path) -> None:
    """Handling FileNotFoundError during parsing/loading."""
    csv_file = tmp_path / "missing.csv"
    pd.DataFrame({"salary": [1000]}).to_csv(csv_file, index=False)

    with patch("backend.app.upload.csv_loader.CSVLoader.load", side_effect=FileNotFoundError("Missing file")):
        with open(csv_file, "rb") as f:
            response = client.post(
                "/api/v1/eda/analyze",
                files={"file": ("missing.csv", f, "text/csv")},
            )

    assert response.status_code == 400
    assert "Missing file" in response.json()["detail"]


def test_analyze_service_failure(client: TestClient, tmp_path: Path) -> None:
    """EDAServiceError should return 400 Bad Request."""
    mock_analysis = MagicMock(spec=AnalysisService)
    mock_eda = MagicMock(spec=EDAService)

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis
    app.dependency_overrides[get_eda_service] = lambda: mock_eda

    mock_eda.run.side_effect = EDAServiceError("Internal pipeline crash")

    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"salary": [1000]}).to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        response = client.post(
            "/api/v1/eda/analyze",
            files={"file": ("data.csv", f, "text/csv")},
        )

    assert response.status_code == 400
    assert "Internal pipeline crash" in response.json()["detail"]


def test_analyze_unexpected_exception(client: TestClient, tmp_path: Path) -> None:
    """Any other unexpected exception should return 500 Internal Server Error."""
    mock_analysis = MagicMock(spec=AnalysisService)
    mock_eda = MagicMock(spec=EDAService)

    app.dependency_overrides[get_analysis_service] = lambda: mock_analysis
    app.dependency_overrides[get_eda_service] = lambda: mock_eda

    mock_eda.run.side_effect = Exception("Wild exception")

    csv_file = tmp_path / "data.csv"
    pd.DataFrame({"salary": [1000]}).to_csv(csv_file, index=False)

    with open(csv_file, "rb") as f:
        response = client.post(
            "/api/v1/eda/analyze",
            files={"file": ("data.csv", f, "text/csv")},
        )

    assert response.status_code == 500
    assert "unexpected error" in response.json()["detail"].lower()


# ── Direct Route Invocation (For 100% Coverage) ────────────────────────────────


def test_analyze_oversized_file_header_direct() -> None:
    """Direct invocation: test that content-length header > 50MB triggers 413 HTTP Exception."""
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.csv"
    mock_file.headers = Headers({"content-length": str(MAX_FILE_SIZE + 1)})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(analyze_dataset(file=mock_file))

    assert exc_info.value.status_code == 413
    assert "File size exceeds the maximum limit of 50 MB" in exc_info.value.detail


def test_analyze_invalid_content_length_header_direct() -> None:
    """Direct invocation: test that malformed content-length header catches ValueError and falls through."""
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.csv"
    mock_file.headers = Headers({"content-length": "not_a_number"})
    mock_file.read.side_effect = ValueError("Stopping early on file read")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(analyze_dataset(file=mock_file))

    # The ValueError during read should bubble up as 400 Bad Request
    assert exc_info.value.status_code == 400
