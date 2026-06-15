"""CSV loading utilities for uploaded datasets."""

from pathlib import Path

import logging
import pandas as pd


logger = logging.getLogger(__name__)


class CSVLoader:
    """Load CSV files into pandas DataFrames with input validation."""

    CSV_EXTENSION = ".csv"

    def load(self, file_path: str) -> pd.DataFrame:
        """Load a CSV file from disk and return it as a non-empty DataFrame.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A pandas DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a CSV or the loaded DataFrame is empty.
            RuntimeError: If pandas fails to read the CSV file.
        """
        path = Path(file_path)
        self._validate_file(path)

        logger.info("Loading CSV file: %s", path)

        try:
            dataframe = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            logger.exception("CSV file has no parseable data: %s", path)
            raise ValueError(f"CSV file is empty or invalid: {path}") from exc
        except pd.errors.ParserError as exc:
            logger.exception("CSV parsing failed for file: %s", path)
            raise RuntimeError(f"Failed to parse CSV file: {path}") from exc
        except OSError as exc:
            logger.exception("Unable to read CSV file: %s", path)
            raise RuntimeError(f"Unable to read CSV file: {path}") from exc

        self._validate_dataframe(dataframe, path)
        logger.info("CSV loaded successfully: %s", path)
        return dataframe

    def _validate_file(self, path: Path) -> None:
        """Validate that the provided path points to an existing CSV file."""
        if not path.exists():
            logger.error("CSV file does not exist: %s", path)
            raise FileNotFoundError(f"CSV file not found: {path}")

        if not path.is_file():
            logger.error("CSV path is not a file: %s", path)
            raise ValueError(f"CSV path must point to a file: {path}")

        if path.suffix.lower() != self.CSV_EXTENSION:
            logger.error("Invalid CSV file extension for path: %s", path)
            raise ValueError(f"File must have a .csv extension: {path}")

    def _validate_dataframe(self, dataframe: pd.DataFrame, path: Path) -> None:
        """Validate that the loaded DataFrame contains data."""
        if dataframe.empty:
            logger.error("Loaded CSV DataFrame is empty: %s", path)
            raise ValueError(f"CSV file contains no rows or columns: {path}")
