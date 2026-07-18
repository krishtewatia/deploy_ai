"""Upload orchestration service for dataset files."""

from pathlib import Path
from typing import Any, Dict, Optional

import logging

from backend.app.upload.csv_loader import CSVLoader
from backend.app.upload.excel_loader import ExcelLoader
from backend.app.upload.metadata_extractor import MetadataExtractor
from backend.app.upload.preview_generator import PreviewGenerator
from backend.app.upload.validator import DatasetValidator


logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file type is not supported."""


class UploadService:
    """Coordinate dataset upload loading, validation, preview, and metadata extraction."""

    CSV_EXTENSION = ".csv"
    EXCEL_EXTENSIONS = {".xlsx", ".xls"}

    def __init__(
        self,
        csv_loader: Optional[CSVLoader] = None,
        excel_loader: Optional[ExcelLoader] = None,
        validator: Optional[DatasetValidator] = None,
        preview_generator: Optional[PreviewGenerator] = None,
        metadata_extractor: Optional[MetadataExtractor] = None,
    ) -> None:
        """Initialize the upload service with injectable dependencies.

        Args:
            csv_loader: Loader used for CSV datasets.
            excel_loader: Loader used for Excel datasets.
            validator: Validator used to verify loaded datasets.
            preview_generator: Generator used to create dataset previews.
            metadata_extractor: Extractor used to create dataset metadata.
        """
        self.csv_loader = csv_loader or CSVLoader()
        self.excel_loader = excel_loader or ExcelLoader()
        self.validator = validator or DatasetValidator()
        self.preview_generator = preview_generator or PreviewGenerator()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

    def process(self, file_path: str, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """Process an uploaded dataset file and return metadata with preview data.

        Args:
            file_path: Path to the uploaded dataset file.
            original_filename: Original name of the uploaded dataset file.

        Returns:
            Dictionary containing dataset metadata and preview payload.

        Raises:
            UnsupportedFileTypeError: If the uploaded file extension is unsupported.
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If loading or validation fails.
            RuntimeError: If a loader fails to read the dataset file.
        """
        path = Path(file_path)
        logger.info("Processing uploaded dataset: %s", path)

        loader = self._get_loader(path)
        dataframe = loader.load(file_path)

        self.validator.validate_dataframe(dataframe)

        metadata_name = original_filename or path.name
        metadata = self.metadata_extractor.extract(dataframe, metadata_name)
        preview = self.preview_generator.generate_preview(dataframe)

        logger.info("Uploaded dataset processed successfully: %s", path)
        return {
            "metadata": metadata,
            "preview": preview,
        }


    def _get_loader(self, path: Path) -> Any:
        """Return the correct dataset loader based on file extension."""
        extension = path.suffix.lower()

        if extension == self.CSV_EXTENSION:
            logger.debug("Detected CSV file type: %s", path)
            return self.csv_loader

        if extension in self.EXCEL_EXTENSIONS:
            logger.debug("Detected Excel file type: %s", path)
            return self.excel_loader

        logger.error("Unsupported uploaded file type: %s", path)
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. Supported types are: "
            f"{self.CSV_EXTENSION}, {', '.join(sorted(self.EXCEL_EXTENSIONS))}."
        )
