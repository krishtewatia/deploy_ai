"""Excel loading utilities for uploaded datasets."""

from pathlib import Path

import logging
import pandas as pd


logger = logging.getLogger(__name__)


class ExcelLoader:
    """Load Excel files into pandas DataFrames with input validation."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls"}

    def load(self, file_path: str) -> pd.DataFrame:
        """Load an Excel file from disk and return it as a non-empty DataFrame.

        Args:
            file_path: Path to the Excel file.

        Returns:
            A pandas DataFrame containing the Excel data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported or the loaded DataFrame is empty.
            RuntimeError: If pandas fails to read the Excel file.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info("Loading Excel file: %s", path)

        try:
            dataframe = pd.read_excel(path)
        except ValueError as exc:
            logger.exception("Excel file is invalid or unsupported: %s", path)
            raise ValueError(f"Excel file is invalid or unsupported: {path}") from exc
        except OSError as exc:
            logger.exception("Unable to read Excel file: %s", path)
            raise RuntimeError(f"Unable to read Excel file: {path}") from exc
        except Exception as exc:
            logger.exception("Unexpected error while loading Excel file: %s", path)
            raise RuntimeError(f"Failed to load Excel file: {path}") from exc

        self._validate_dataframe(dataframe, path)
        logger.info("Excel file loaded successfully: %s", path)
        return dataframe

    def _validate_file(self, path: Path) -> None:
        """Validate that the provided path points to a supported Excel file."""
        if not path.exists():
            logger.error("Excel file does not exist: %s", path)
            raise FileNotFoundError(f"Excel file not found: {path}")

        if not path.is_file():
            logger.error("Excel path is not a file: %s", path)
            raise ValueError(f"Excel path must point to a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.error("Unsupported Excel file extension for path: %s", path)
            raise ValueError(
                f"File must have one of these extensions: "
                f"{', '.join(sorted(self.SUPPORTED_EXTENSIONS))}: {path}"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame, path: Path) -> None:
        """Validate that the loaded DataFrame contains data."""
        if dataframe.empty:
            logger.error("Loaded Excel DataFrame is empty: %s", path)
            raise ValueError(f"Excel file contains no rows or columns: {path}")
