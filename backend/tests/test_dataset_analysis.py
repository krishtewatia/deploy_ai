"""Unit and integration tests for the Dataset Analysis Dashboard (Stage 12F)."""

import os
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from PySide6.QtWidgets import QMessageBox, QDialog
from PySide6.QtCore import Qt
from frontend.app import DeployAIApplication
from frontend.pages.dataset_analysis import DatasetAnalysisWidget


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def _make_mock_metadata(name: str, location: str, size: int = 100) -> dict:
    return {
        "name": name,
        "file_name": os.path.basename(location),
        "file_type": "CSV",
        "rows": 10,
        "columns": 3,
        "size_bytes": size,
        "imported_time": "2026-07-15T12:00:00Z",
        "location": location,
        "missing_values": 0,
        "duplicate_rows": 0,
        "target": "target_col",
        "status": "Active",
        "column_names": ["col1", "col2", "target_col"]
    }


class TestDatasetAnalysisWidget:

    def test_widget_initial_state(self) -> None:
        """Verify the widget fields and labels start with default values."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        # Check default summary labels
        assert widget.lbl_rows.text() == "-"
        assert widget.lbl_cols.text() == "-"
        assert widget.lbl_mem.text() == "-"
        assert widget.lbl_file_size.text() == "-"
        assert widget.lbl_target.text() == "-"
        assert widget.lbl_status.text() == "-"

        # Check default quality labels
        assert widget.lbl_missing.text() == "-"
        assert widget.lbl_dup_rows.text() == "-"
        assert widget.lbl_dup_cols.text() == "-"
        assert widget.lbl_const_cols.text() == "-"
        assert widget.lbl_high_card.text() == "-"

        # Check default readiness labels
        assert widget.lbl_readiness_score.text() == "0%"
        assert widget.lbl_readiness_level.text() == "-"
        assert widget.progress_bar.value() == 0
        assert widget.table.rowCount() == 0

    def test_set_dataset_missing_file(self, tmp_path: Path) -> None:
        """Verify widget warns and triggers refresh if the dataset file does not exist on disk."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        meta = _make_mock_metadata("Missing", str(tmp_path / "missing.csv"))

        parent_stack = MagicMock()
        parent_manager = MagicMock()
        parent_stack.parent.return_value = parent_manager
        with patch.object(widget, "parent", return_value=parent_stack):
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
                widget.set_dataset(meta)
                mock_warn.assert_called_with(widget, "Dataset Error", "The selected dataset file does not exist on disk.")
                parent_manager.refresh_datasets.assert_called_once()


    def test_set_dataset_corrupted_metadata(self, tmp_path: Path) -> None:
        """Verify widget warns user if metadata fields are incomplete or corrupted."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        target_file = tmp_path / "valid.csv"
        target_file.touch()

        # Incomplete metadata dictionary missing "file_name"
        meta = {
            "name": "Corrupt",
            "location": str(target_file)
        }

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            widget.set_dataset(meta)
            mock_warn.assert_called_with(widget, "Metadata Error", "The dataset metadata is corrupted or incomplete.")

    def test_set_dataset_pandas_read_error(self, tmp_path: Path) -> None:
        """Verify loader exceptions are caught and reported as critical analysis failures."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        target_file = tmp_path / "broken.csv"
        target_file.touch()

        meta = _make_mock_metadata("Broken", str(target_file))

        with patch("pandas.read_csv", side_effect=ValueError("pandas crash")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                widget.set_dataset(meta)
                mock_crit.assert_called_with(widget, "Analysis Error", "Failed to perform dataset analysis: pandas crash")

    def test_set_dataset_success_csv(self, tmp_path: Path) -> None:
        """Verify successful CSV load, profiling, and widget populating."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        target_file = tmp_path / "data.csv"
        # 5 rows, col1 has duplicate/missing, target has categories, col2 is constant
        df = pd.DataFrame({
            "col1": [1, 2, None, 2, 5],
            "col2": ["constant", "constant", "constant", "constant", "constant"],
            "target": ["A", "B", "A", "B", "A"]
        })
        df.to_csv(target_file, index=False)

        meta = _make_mock_metadata("My CSV", str(target_file), size=200)

        widget.set_dataset(meta)

        # Check Summary
        assert widget.lbl_rows.text() == "5"
        assert widget.lbl_cols.text() == "3"
        assert widget.lbl_file_size.text() == "200 B"
        assert widget.lbl_target.text() == "target_col"
        assert widget.lbl_status.text() == "Active"

        # Check Quality
        assert widget.lbl_missing.text() == "1"
        assert widget.lbl_dup_rows.text() == "1 (20.0%)"
        assert widget.lbl_const_cols.text() == "1"  # col2 is constant

        # Check Table
        assert widget.table.rowCount() == 3
        # col1 check
        assert widget.table.item(0, 0).text() == "col1"
        assert widget.table.item(0, 2).text() == "1 (20.0%)" # 1 missing
        assert widget.table.item(0, 3).text() == "3 (60.0%)" # 3 unique non-null values (1.0, 2.0, 5.0)

        # Check Readiness
        assert widget.progress_bar.value() > 0
        assert widget.lbl_readiness_level.text() in ["Excellent", "Good", "Poor"]

    def test_set_dataset_success_excel(self, tmp_path: Path) -> None:
        """Verify successful Excel load using pandas read_excel mock."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        target_file = tmp_path / "data.xlsx"
        target_file.touch()

        meta = _make_mock_metadata("My Excel", str(target_file))

        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6]
        })

        with patch("pandas.read_excel", return_value=df):
            widget.set_dataset(meta)

        assert widget.lbl_rows.text() == "3"
        assert widget.lbl_cols.text() == "2"

    def test_on_refresh_errors(self, tmp_path: Path) -> None:
        """Verify refresh warning dialogs on missing or deleted files."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        # Case 1: No current metadata selected
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            widget._on_refresh()
            mock_warn.assert_called_with(widget, "Refresh Error", "No dataset is currently selected.")

        # Case 2: Metadata file location doesn't exist
        meta = _make_mock_metadata("Missing", str(tmp_path / "missing.csv"))
        widget.current_metadata = meta
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            widget._on_refresh()
            mock_warn.assert_called_with(widget, "Dataset Error", "The selected dataset file does not exist on disk.")

        # Case 3: Pandas read exception during refresh
        target_file = tmp_path / "valid.csv"
        target_file.touch()
        meta = _make_mock_metadata("Valid", str(target_file))
        widget.current_metadata = meta

        with patch("pandas.read_csv", side_effect=RuntimeError("refresh pandas crash")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                widget._on_refresh()
                mock_crit.assert_called_with(widget, "Refresh Error", "Failed to refresh dataset analysis: refresh pandas crash")

    def test_on_refresh_success(self, tmp_path: Path) -> None:
        """Verify refresh successfully re-reads dataset, updates UI, and writes to metadata JSON."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        # Create parent widget mock structure to simulate parent manager refresh
        parent_stack = MagicMock()
        parent_manager = MagicMock()
        parent_stack.parent.return_value = parent_manager
        with patch.object(widget, "parent", return_value=parent_stack):
            
            target_file = tmp_path / "refresh_data.csv"
            df_initial = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            df_initial.to_csv(target_file, index=False)

            meta = _make_mock_metadata("RefreshTest", str(target_file))
            meta_json_path = target_file.with_suffix(target_file.suffix + ".json")
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            widget.set_dataset(meta)
            assert widget.lbl_rows.text() == "2"

            # Modify CSV contents (add 2 rows)
            df_new = pd.DataFrame({"a": [1, 2, 5, 6], "b": [3, 4, 7, 8]})
            df_new.to_csv(target_file, index=False)

            with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
                widget._on_refresh()
                mock_info.assert_called_with(widget, "Success", "Dataset analysis refreshed successfully.")

            # UI summary labels should be updated
            assert widget.lbl_rows.text() == "4"

            # Metadata JSON file should be updated on disk
            with open(meta_json_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
                assert saved_meta["rows"] == 4

            # Parent manager should have been refreshed and item re-selected
            parent_manager.refresh_datasets.assert_called_once()
            parent_manager.select_dataset_by_name.assert_called_with("RefreshTest")

    def test_on_refresh_excel_success(self, tmp_path: Path) -> None:
        """Verify refresh successfully re-reads Excel file and invokes pd.read_excel."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        # Create parent widget mock structure
        parent_stack = MagicMock()
        parent_manager = MagicMock()
        parent_stack.parent.return_value = parent_manager
        with patch.object(widget, "parent", return_value=parent_stack):
            target_file = tmp_path / "refresh_data.xlsx"
            target_file.touch()

            meta = _make_mock_metadata("RefreshExcel", str(target_file))
            meta["file_type"] = "Excel"
            meta_json_path = target_file.with_suffix(target_file.suffix + ".json")
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)

            df_initial = pd.DataFrame({"a": [1, 2]})
            
            with patch("pandas.read_excel", return_value=df_initial):
                widget.set_dataset(meta)
                assert widget.lbl_rows.text() == "2"

            df_new = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
            with patch("pandas.read_excel", return_value=df_new):
                with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
                    widget._on_refresh()
                    mock_info.assert_called_with(widget, "Success", "Dataset analysis refreshed successfully.")
                    assert widget.lbl_rows.text() == "5"

    def test_readiness_score_formulas(self, tmp_path: Path) -> None:
        """Verify readiness deduction thresholds for missing cells, duplicates, and cardinality."""
        app_obj = DeployAIApplication()
        widget = DatasetAnalysisWidget()

        target_file = tmp_path / "extreme.csv"
        # 1. Test clean dataset (score 100)
        df_clean = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [10, 20, 30, 40, 50]
        })
        df_clean.to_csv(target_file, index=False)
        meta = _make_mock_metadata("Clean", str(target_file))
        widget.set_dataset(meta)
        assert widget.progress_bar.value() == 100
        assert widget.lbl_readiness_level.text() == "Excellent"

        # 2. Test high cardinality and duplicate columns/constant columns dataset (Yellow or Red)
        # col2 constant, col3 duplicated from col1, col1 categorical with high cardinality
        df_dirty = pd.DataFrame({
            "col1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v"],
            "col2": [1] * 22,
            "col3": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v"]
        })
        df_dirty.to_csv(target_file, index=False)
        meta_dirty = _make_mock_metadata("Dirty", str(target_file))
        widget.set_dataset(meta_dirty)
        
        # Verify score is lower
        assert widget.progress_bar.value() < 90

        # 3. Test clean dataset with >= 100 rows to check if not warnings (no warnings path)
        df_large_clean = pd.DataFrame({
            "col1": list(range(120)),
            "col2": list(range(100, 220))
        })
        large_file = tmp_path / "large_clean.csv"
        df_large_clean.to_csv(large_file, index=False)
        meta_large = _make_mock_metadata("LargeClean", str(large_file))
        widget.set_dataset(meta_large)
        assert "No significant warnings" in widget.txt_warnings.toPlainText()
        assert widget.progress_bar.value() == 100

        # 4. Test very dirty dataset with Poor readiness score (< 50)
        # 100 rows, col1 is constant, col2 constant, col3 constant, col4 constant, col5 constant, col6 high cardinality, col7 constant, etc.
        df_poor = pd.DataFrame({
            "col1": [1] * 120,
            "col2": [2] * 120,
            "col3": [3] * 120,
            "col4": [4] * 120,
            "col5": [5] * 120,
            "col6": [str(x) for x in range(120)],
            "col7": [6] * 120,
        })
        # Add duplicate rows
        df_poor.iloc[50:110] = df_poor.iloc[0:60]
        # Add missing values
        df_poor.iloc[0:30, 0:5] = None

        poor_file = tmp_path / "poor.csv"
        df_poor.to_csv(poor_file, index=False)
        meta_poor = _make_mock_metadata("PoorData", str(poor_file))
        
        widget.set_dataset(meta_poor)
        assert widget.progress_bar.value() < 50
        assert widget.lbl_readiness_level.text() == "Poor"

