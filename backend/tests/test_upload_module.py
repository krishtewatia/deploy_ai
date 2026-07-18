"""Tests for dataset upload utilities and orchestration."""

from pathlib import Path

import pandas as pd
import pytest

from backend.app.upload.csv_loader import CSVLoader
from backend.app.upload.excel_loader import ExcelLoader
from backend.app.upload.metadata_extractor import MetadataExtractor
from backend.app.upload.preview_generator import PreviewGenerator
from backend.app.upload.schemas import DatasetMetadata
from backend.app.upload.upload_service import UnsupportedFileTypeError, UploadService
from backend.app.upload.validator import (
    DuplicateColumnNamesError,
    EmptyDataFrameError,
    MissingColumnsError,
    DatasetValidator,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a representative mixed-type DataFrame for upload tests."""
    return pd.DataFrame(
        {
            "age": [25, 31],
            "name": ["Ada", "Grace"],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )


def test_csv_loader_loads_valid_csv(tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """CSVLoader should load a valid CSV file into a DataFrame."""
    file_path = tmp_path / "dataset.csv"
    sample_dataframe.to_csv(file_path, index=False)

    loaded_dataframe = CSVLoader().load(str(file_path))

    assert loaded_dataframe.shape == (2, 3)
    assert loaded_dataframe.columns.tolist() == ["age", "name", "created_at"]


def test_csv_loader_rejects_missing_file(tmp_path: Path) -> None:
    """CSVLoader should reject missing files with a clear error."""
    with pytest.raises(FileNotFoundError, match="CSV file not found"):
        CSVLoader().load(str(tmp_path / "missing.csv"))


def test_csv_loader_rejects_non_csv_file(tmp_path: Path) -> None:
    """CSVLoader should reject files without the .csv extension."""
    file_path = tmp_path / "dataset.txt"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.csv extension"):
        CSVLoader().load(str(file_path))


def test_csv_loader_rejects_empty_csv(tmp_path: Path) -> None:
    """CSVLoader should reject empty CSV files."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty or invalid"):
        CSVLoader().load(str(file_path))


def test_excel_loader_loads_valid_xlsx(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExcelLoader should load a valid XLSX file into a DataFrame."""
    file_path = tmp_path / "dataset.xlsx"
    file_path.write_bytes(b"placeholder")

    def fake_read_excel(path: Path) -> pd.DataFrame:
        assert path == file_path
        return sample_dataframe

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    loaded_dataframe = ExcelLoader().load(str(file_path))

    assert loaded_dataframe.equals(sample_dataframe)


def test_excel_loader_supports_xls_extension(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExcelLoader should support legacy .xls files."""
    file_path = tmp_path / "dataset.xls"
    file_path.write_bytes(b"placeholder")
    monkeypatch.setattr(pd, "read_excel", lambda path: sample_dataframe)

    loaded_dataframe = ExcelLoader().load(str(file_path))

    assert loaded_dataframe.equals(sample_dataframe)


def test_excel_loader_rejects_missing_file(tmp_path: Path) -> None:
    """ExcelLoader should reject missing files."""
    with pytest.raises(FileNotFoundError, match="Excel file not found"):
        ExcelLoader().load(str(tmp_path / "missing.xlsx"))


def test_excel_loader_rejects_unsupported_extension(tmp_path: Path) -> None:
    """ExcelLoader should reject unsupported file extensions."""
    file_path = tmp_path / "dataset.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extensions"):
        ExcelLoader().load(str(file_path))


def test_excel_loader_rejects_empty_dataframe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ExcelLoader should reject Excel files that load into empty DataFrames."""
    file_path = tmp_path / "empty.xlsx"
    file_path.write_bytes(b"placeholder")
    monkeypatch.setattr(pd, "read_excel", lambda path: pd.DataFrame())

    with pytest.raises(ValueError, match="no rows or columns"):
        ExcelLoader().load(str(file_path))


def test_dataset_validator_accepts_valid_dataframe(sample_dataframe: pd.DataFrame) -> None:
    """DatasetValidator should accept valid DataFrames."""
    DatasetValidator().validate_dataframe(sample_dataframe)


def test_dataset_validator_rejects_empty_dataframe() -> None:
    """DatasetValidator should reject empty DataFrames."""
    with pytest.raises(EmptyDataFrameError, match="must not be empty"):
        DatasetValidator().validate_dataframe(pd.DataFrame())


def test_dataset_validator_rejects_dataframe_without_columns() -> None:
    """DatasetValidator should reject DataFrames without columns."""
    dataframe = pd.DataFrame(index=[0, 1])

    with pytest.raises(MissingColumnsError, match="at least one column"):
        DatasetValidator().validate_dataframe(dataframe)


def test_dataset_validator_rejects_duplicate_column_names() -> None:
    """DatasetValidator should reject duplicate column names."""
    dataframe = pd.DataFrame([[1, 2]], columns=["value", "value"])

    with pytest.raises(DuplicateColumnNamesError, match="Duplicate column names"):
        DatasetValidator().validate_dataframe(dataframe)


def test_metadata_extractor_returns_dataset_metadata(sample_dataframe: pd.DataFrame) -> None:
    """MetadataExtractor should classify columns and return DatasetMetadata."""
    metadata = MetadataExtractor().extract(sample_dataframe, "dataset.csv")

    assert isinstance(metadata, DatasetMetadata)
    assert metadata.file_name == "dataset.csv"
    assert metadata.rows == 2
    assert metadata.columns == 3
    assert metadata.column_names == ["age", "name", "created_at"]
    assert metadata.numerical_columns == ["age"]
    assert metadata.categorical_columns == ["name"]
    assert metadata.datetime_columns == ["created_at"]
    assert metadata.target_column is None


def test_preview_generator_returns_shape_columns_and_records(
    sample_dataframe: pd.DataFrame,
) -> None:
    """PreviewGenerator should return shape, columns, and row records."""
    preview = PreviewGenerator().generate_preview(sample_dataframe, rows=1)

    assert preview["shape"] == [2, 3]
    assert preview["columns"] == ["age", "name", "created_at"]
    assert len(preview["preview"]) == 1
    assert preview["preview"][0]["age"] == 25


def test_upload_service_processes_csv_upload(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """UploadService should load, validate, preview, and extract metadata for CSV files."""
    file_path = tmp_path / "dataset.csv"
    sample_dataframe.to_csv(file_path, index=False)

    result = UploadService().process(str(file_path))

    assert isinstance(result["metadata"], DatasetMetadata)
    assert result["metadata"].file_name == "dataset.csv"
    assert result["metadata"].rows == 2
    assert result["preview"]["shape"] == [2, 3]
    assert len(result["preview"]["preview"]) == 2


def test_upload_service_uses_injected_loader_and_collaborators(
    sample_dataframe: pd.DataFrame,
) -> None:
    """UploadService should support dependency injection for collaborators."""

    class FakeLoader:
        def load(self, file_path: str) -> pd.DataFrame:
            assert file_path == "dataset.csv"
            return sample_dataframe

    class FakeValidator:
        def __init__(self) -> None:
            self.called = False

        def validate_dataframe(self, df: pd.DataFrame) -> None:
            assert df is sample_dataframe
            self.called = True

    class FakePreviewGenerator:
        def generate_preview(self, df: pd.DataFrame) -> dict[str, object]:
            assert df is sample_dataframe
            return {"shape": [2, 3], "columns": [], "preview": []}

    class FakeMetadataExtractor:
        def extract(self, df: pd.DataFrame, file_name: str) -> DatasetMetadata:
            assert df is sample_dataframe
            assert file_name == "dataset.csv"
            return DatasetMetadata(
                file_name=file_name,
                rows=2,
                columns=3,
                column_names=[],
                numerical_columns=[],
                categorical_columns=[],
                datetime_columns=[],
                target_column=None,
            )

    validator = FakeValidator()
    service = UploadService(
        csv_loader=FakeLoader(),
        validator=validator,
        preview_generator=FakePreviewGenerator(),
        metadata_extractor=FakeMetadataExtractor(),
    )

    result = service.process("dataset.csv")

    assert validator.called is True
    assert result["metadata"].file_name == "dataset.csv"
    assert result["preview"]["shape"] == [2, 3]


def test_upload_service_rejects_unsupported_file_type() -> None:
    """UploadService should reject unsupported file extensions."""
    with pytest.raises(UnsupportedFileTypeError, match="Unsupported file type"):
        UploadService().process("dataset.json")

