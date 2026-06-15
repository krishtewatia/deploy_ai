"""Preview generation utilities for uploaded datasets."""

from typing import Any, Dict

import pandas as pd


class PreviewGenerator:
    """Generate lightweight previews for uploaded dataset DataFrames."""

    def generate_preview(self, df: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
        """Generate a serializable preview payload from a DataFrame.

        Args:
            df: DataFrame to preview.
            rows: Maximum number of rows to include in the preview.

        Returns:
            Dictionary containing the DataFrame shape, column names, and preview rows.
        """
        return {
            "shape": [df.shape[0], df.shape[1]],
            "columns": df.columns.tolist(),
            "preview": df.head(rows).to_dict(orient="records"),
        }
