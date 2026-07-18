"""Per-column profiling for DataFrames.

Produces a pure-Python dictionary with dtype, uniqueness, missingness,
sample values, and type-category flags for every column — ready to be
consumed by the AI Recommendation Engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Maximum number of unique non-null sample values per column.
_MAX_SAMPLE_VALUES: int = 5


class ColumnProfiler:
    """Generate detailed per-column metadata from a :class:`~pandas.DataFrame`.

    Usage::

        profiler = ColumnProfiler()
        profiles = profiler.profile(df)
        # profiles["age"]["is_numeric"]  →  True
    """

    # ── public API ──────────────────────────────────────────────────────

    def profile(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Return a profiling dictionary keyed by column name.

        Parameters
        ----------
        df:
            The DataFrame to profile.

        Returns
        -------
        dict
            ``{ column_name: { dtype, unique_values, unique_percentage,
            missing_count, missing_percentage, sample_values, is_numeric,
            is_categorical, is_datetime } }``

            All percentages are rounded to two decimal places.
            ``sample_values`` contains at most 5 unique non-null examples.
        """
        logger.info("Profiling %d columns across %d rows.", len(df.columns), len(df))

        result: Dict[str, Dict[str, Any]] = {}

        for col in df.columns:
            result[str(col)] = self._profile_column(df[col], len(df))

        logger.info("Column profiling complete.")
        return result

    # ── private helpers ─────────────────────────────────────────────────

    @staticmethod
    def _profile_column(series: pd.Series, total_rows: int) -> Dict[str, Any]:
        """Build the profile dict for a single *series*."""
        missing_count: int = int(series.isna().sum())
        unique_count: int = int(series.nunique(dropna=True))

        missing_pct = round(missing_count / total_rows * 100, 2) if total_rows > 0 else 0.0
        unique_pct = round(unique_count / total_rows * 100, 2) if total_rows > 0 else 0.0

        sample_values = ColumnProfiler._get_sample_values(series)

        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        is_categorical = (
            isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_bool_dtype(series)
        ) and not is_datetime

        return {
            "dtype": str(series.dtype),
            "unique_values": unique_count,
            "unique_percentage": unique_pct,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "sample_values": sample_values,
            "is_numeric": is_numeric,
            "is_categorical": is_categorical,
            "is_datetime": is_datetime,
        }

    @staticmethod
    def _get_sample_values(series: pd.Series) -> List[Any]:
        """Return up to :data:`_MAX_SAMPLE_VALUES` unique non-null values.

        Values are converted to native Python types so the result is
        JSON-serialisable.
        """
        unique_vals = series.dropna().unique()[:_MAX_SAMPLE_VALUES]
        samples: List[Any] = []
        for v in unique_vals:
            if hasattr(v, "isoformat"):
                samples.append(v.isoformat())
            else:
                try:
                    samples.append(v.item())  # numpy scalar → Python native
                except AttributeError:
                    samples.append(v)
        return samples
