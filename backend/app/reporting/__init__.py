"""Reporting package.

Provides automated packaging of final model development logs, AI critiques,
optimization operations, and governance decisions into user-facing executive reports.
"""

from backend.app.reporting.schemas import ExecutiveReport
from backend.app.reporting.report_generator import (
    ExecutiveReportGenerator,
    ExecutiveReportGeneratorError,
)

__all__ = [
    "ExecutiveReport",
    "ExecutiveReportGenerator",
    "ExecutiveReportGeneratorError",
]
