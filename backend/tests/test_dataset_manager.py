"""Unit and integration tests for the Dataset Manager Page and Import Dataset Dialog (Stage 12E)."""

import os
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from PySide6.QtWidgets import QDialog, QMessageBox, QTableWidgetItem
from PySide6.QtCore import Qt, QPoint
from frontend.app import DeployAIApplication
from frontend.pages.dataset_manager import DatasetManagerPage, format_size, format_timestamp
from frontend.pages.dialogs.import_dataset_dialog import ImportDatasetDialog


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def _make_mock_metadata(name: str = "Train Data", file_name: str = "train.csv", size: int = 2048) -> dict:
    return {
        "name": name,
        "file_name": file_name,
        "file_type": "CSV",
        "rows": 150,
        "columns": 5,
        "size_bytes": size,
        "imported_time": "2026-07-15T12:00:00Z",
        "location": f"/mock/workspace/datasets/{file_name}",
        "missing_values": 10,
        "duplicate_rows": 2,
        "target": "target_col",
        "status": "Active",
        "column_names": ["c1", "c2", "c3", "c4", "target_col"]
    }


# =======================================================================
# Helpers & Size/Timestamp Formatting Tests
# =======================================================================

def test_size_formatting() -> None:
    """Verify bytes are correctly formatted into human-readable size labels."""
    assert format_size(500) == "500 B"
    assert format_size(1024) == "1.0 KB"
    assert format_size(1536) == "1.5 KB"
    assert format_size(1048576) == "1.0 MB"
    assert format_size(2097152) == "2.0 MB"


def test_timestamp_formatting() -> None:
    """Verify ISO timestamp strings are correctly formatted for display."""
    iso_time = "2026-07-15T12:30:45Z"
    assert "2026-07-15 12:30:45 UTC" in format_timestamp(iso_time)
    assert format_timestamp("invalid-date") == "invalid-date"


# =======================================================================
# ImportDatasetDialog Tests
# =======================================================================

class TestImportDatasetDialog:

    def test_dialog_fields_initial_state(self, tmp_path: Path) -> None:
        """Verify the dialog fields, buttons, and browse setup are instantiated properly."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        assert dialog.txt_file.text() == ""
        assert dialog.txt_name.text() == ""
        assert dialog.btn_browse is not None
        assert dialog.btn_import is not None
        assert dialog.btn_cancel is not None
        assert dialog.imported_metadata_path is None

    def test_dialog_browse_button(self, tmp_path: Path) -> None:
        """Verify clicking browse opens QFileDialog and autofills dataset name from basename."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        
        target_file = tmp_path / "my_custom_data.csv"
        target_file.touch()

        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileName", return_value=(str(target_file), "csv")):
            dialog._on_browse()
            assert dialog.txt_file.text() == str(target_file)
            assert dialog.txt_name.text() == "my_custom_data"

    def test_dialog_validation_workspace_closed(self) -> None:
        """Verify import fails immediately with warning if workspace_path is invalid."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path="/invalid/path/that/does/not/exist")
        dialog.txt_file.setText("/some/file.csv")
        dialog.txt_name.setText("TestName")

        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            dialog._on_import()
            mock_crit.assert_called_with(dialog, "Import Error", "No active workspace is open.")

    def test_dialog_validation_empty_inputs(self, tmp_path: Path) -> None:
        """Verify import warns user if filename or name inputs are empty."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            # Case 1: Empty file
            dialog._on_import()
            mock_warn.assert_called_with(dialog, "Validation Error", "Please select a dataset file.")

            # Case 2: Empty name
            mock_warn.reset_mock()
            dialog.txt_file.setText("some_file.csv")
            dialog._on_import()
            mock_warn.assert_called_with(dialog, "Validation Error", "Please enter a dataset name.")

    def test_dialog_validation_file_not_found(self, tmp_path: Path) -> None:
        """Verify warning shown if specified file path does not exist."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        dialog.txt_file.setText(str(tmp_path / "does_not_exist.csv"))
        dialog.txt_name.setText("Name")

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_import()
            mock_warn.assert_called_with(dialog, "Validation Error", "Selected file does not exist.")

    def test_dialog_validation_unsupported_extension(self, tmp_path: Path) -> None:
        """Verify file type validation rejects unsupported suffixes like .txt."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        
        target = tmp_path / "bad.txt"
        target.touch()
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("BadData")

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_import()
            mock_warn.assert_called_with(dialog, "Validation Error", "Unsupported file extension. Only CSV and Excel are supported.")

    def test_dialog_validation_duplicate_name(self, tmp_path: Path) -> None:
        """Verify dataset import fails if dataset name already registered in workspace."""
        app_obj = DeployAIApplication()
        # Create datasets folder and existing metadata
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        existing_meta = datasets_dir / "data1.csv.json"
        
        with open(existing_meta, "w") as f:
            json.dump({"name": "DuplicateName"}, f)

        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target_file = tmp_path / "new.csv"
        target_file.touch()
        dialog.txt_file.setText(str(target_file))
        dialog.txt_name.setText("DuplicateName")

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_import()
            mock_warn.assert_called_with(dialog, "Validation Error", "A dataset named 'DuplicateName' already exists in this project.")

    def test_dialog_validation_unreadable_file(self, tmp_path: Path) -> None:
        """Verify crash or read failures are caught and reported as critical load errors."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "broken.csv"
        target.write_text("a,b,c\n1,2,3", encoding="utf-8")
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("Broken")

        with patch("pandas.read_csv", side_effect=ValueError("pandas crash")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                dialog._on_import()
                mock_crit.assert_called_with(dialog, "Import Error", "Failed to read file: pandas crash")

    def test_dialog_validation_empty_file(self, tmp_path: Path) -> None:
        """Verify empty dataframe shapes are caught and rejected."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "empty.csv"
        target.write_text("a,b\n", encoding="utf-8") # Empty headers only
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("Empty")

        # Mock pandas returning empty df
        empty_df = pd.DataFrame()
        with patch("pandas.read_csv", return_value=empty_df):
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
                dialog._on_import()
                mock_warn.assert_called_with(dialog, "Validation Error", "The dataset file is empty (contains no rows or columns).")

    def test_dialog_validation_duplicate_columns(self, tmp_path: Path) -> None:
        """Verify headers containing duplicate column names are rejected."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "dup.csv"
        target.write_text("a,b,a\n1,2,3", encoding="utf-8")
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("DupHeaders")

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_import()
            mock_warn.assert_called_with(dialog, "Validation Error", "Duplicate column names are not allowed: ['a']")

    def test_dialog_success_import_csv(self, tmp_path: Path) -> None:
        """Verify successful CSV import flow and metadata JSON persistence."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "valid.csv"
        target.write_text("age,name,target\n25,Alice,1\n30,Bob,0\n25,Alice,1", encoding="utf-8")
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("Valid CSV")

        dialog._on_import()
        
        # Copied file check
        copied_file = tmp_path / "datasets" / "valid.csv"
        assert copied_file.exists()
        
        # Metadata check
        meta_file = tmp_path / "datasets" / "valid.csv.json"
        assert meta_file.exists()
        
        with open(meta_file, "r") as f:
            meta = json.load(f)
            assert meta["name"] == "Valid CSV"
            assert meta["rows"] == 3
            assert meta["columns"] == 3
            assert meta["missing_values"] == 0
            assert meta["duplicate_rows"] == 1
            assert meta["column_names"] == ["age", "name", "target"]
            assert meta["status"] == "Active"

    def test_dialog_success_import_excel(self, tmp_path: Path) -> None:
        """Verify successful Excel import flow using pandas monkeypatching."""
        app_obj = DeployAIApplication()
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "valid.xlsx"
        target.write_bytes(b"placeholder")
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("Valid Excel")

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        with patch("pandas.read_excel", return_value=mock_df):
            dialog._on_import()
            
        copied_file = tmp_path / "datasets" / "valid.xlsx"
        assert copied_file.exists()
        
        meta_file = tmp_path / "datasets" / "valid.xlsx.json"
        assert meta_file.exists()

    def test_dialog_filename_collision_handling(self, tmp_path: Path) -> None:
        """Verify filename collision renames the destination file safely using increments."""
        app_obj = DeployAIApplication()
        # Create initial pre-existing file in workspace
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        (datasets_dir / "valid.csv").write_text("orig", encoding="utf-8")

        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "valid.csv"
        target.write_text("age,name\n25,Alice\n30,Bob", encoding="utf-8")
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("Import Two")

        dialog._on_import()

        # Destination file should be renamed to valid_1.csv
        copied_file = tmp_path / "datasets" / "valid_1.csv"
        assert copied_file.exists()
        assert (datasets_dir / "valid.csv").read_text(encoding="utf-8") == "orig"

        # Check metadata points to valid_1.csv
        meta_file = tmp_path / "datasets" / "valid_1.csv.json"
        assert meta_file.exists()
        with open(meta_file, "r") as f:
            meta = json.load(f)
            assert meta["file_name"] == "valid_1.csv"



# =======================================================================
# DatasetManagerPage Tests
# =======================================================================

class TestDatasetManagerPage:

    def test_page_initial_state_no_workspace(self) -> None:
        """Verify controls are disabled and placeholder text shown if no project is active."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.active_workspace_path = None
        
        page = DatasetManagerPage(orchestrator=mock_orch)
        
        assert not page.btn_import.isEnabled()
        assert page.empty_details_label.text() == "Please open or create a project workspace first."
        assert page.details_stack.currentIndex() == 0

    def test_page_initial_state_with_workspace(self, tmp_path: Path) -> None:
        """Verify import buttons enable when a workspace path is active."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path)
        
        page = DatasetManagerPage(orchestrator=mock_orch)
        
        assert page.btn_import.isEnabled()
        assert page.empty_details_label.text() == "No dataset selected"

    def test_page_load_and_refresh_datasets(self, tmp_path: Path) -> None:
        """Verify refresh scans workspace and correctly populates the QTableWidget."""
        app_obj = DeployAIApplication()
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        
        # Save two mock dataset metadata records
        with open(datasets_dir / "ds1.csv.json", "w") as f:
            json.dump(_make_mock_metadata("Data One", "ds1.csv", 1024), f)
        with open(datasets_dir / "ds2.csv.json", "w") as f:
            json.dump(_make_mock_metadata("Data Two", "ds2.csv", 2048000), f)

        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path)

        page = DatasetManagerPage(orchestrator=mock_orch)
        page.refresh_datasets()

        assert page.table.rowCount() == 2
        
        # Validate table row contents
        assert page.table.item(0, 0).text() == "Data One"
        assert page.table.item(0, 1).text() == "CSV"
        assert page.table.item(0, 4).text() == "1.0 KB"

        assert page.table.item(1, 0).text() == "Data Two"
        assert page.table.item(1, 4).text() == "2.0 MB"

    def test_selection_changed_populates_details(self, tmp_path: Path) -> None:
        """Verify selecting a row updates details fields and toggles details stack."""
        app_obj = DeployAIApplication()
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        
        csv_file = datasets_dir / "ds1.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "target_col": [0, 1]})
        df.to_csv(csv_file, index=False)

        meta = _make_mock_metadata("My Dataset", csv_file.name, 5000)
        meta["location"] = str(csv_file)
        with open(datasets_dir / "ds1.csv.json", "w") as f:
            json.dump(meta, f)

        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path)

        page = DatasetManagerPage(orchestrator=mock_orch)
        page.refresh_datasets()

        # Selection empty initially
        assert page.details_stack.currentIndex() == 0

        # Programmatic select row 0
        page.table.selectRow(0)
        page._on_selection_changed()

        # Stack should show details panel and populate labels on the details_widget
        assert page.details_stack.currentIndex() == 1
        assert page.details_widget.lbl_rows.text() == "2"
        assert page.details_widget.lbl_cols.text() == "3"
        assert page.details_widget.lbl_target.text() == "target_col"
        assert page.details_widget.lbl_status.text() == "Active"

        # Deselect should switch back to placeholder
        page.table.clearSelection()
        page._on_selection_changed()
        assert page.details_stack.currentIndex() == 0

    def test_open_import_dialog_success(self, tmp_path: Path) -> None:
        """Verify successful dialog accept triggers listing refresh and autoselects the item."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path)
        
        page = DatasetManagerPage(orchestrator=mock_orch)
        
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = datasets_dir / "imported.csv"
        df = pd.DataFrame({"col1": [1, 2], "target": [0, 1]})
        df.to_csv(csv_file, index=False)

        meta_file = datasets_dir / "imported.csv.json"
        meta = _make_mock_metadata("Imported Data", csv_file.name)
        meta["location"] = str(csv_file)
        with open(meta_file, "w") as f:
            json.dump(meta, f)

        with patch("frontend.pages.dataset_manager.ImportDatasetDialog") as MockDialog:
            dialog_instance = MockDialog.return_value
            dialog_instance.exec.return_value = QDialog.DialogCode.Accepted
            dialog_instance.imported_metadata_path = str(meta_file.resolve())

            page._on_import_dataset()
            
            # Check table contains the new item and it is selected
            assert page.table.rowCount() == 1
            assert page.table.item(0, 0).text() == "Imported Data"
            assert page.table.currentRow() == 0

    def test_show_event_trigger(self, tmp_path: Path) -> None:
        """Verify showEvent calls state updates and listing refresh."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path)
        
        page = DatasetManagerPage(orchestrator=mock_orch)
        
        with patch.object(page, "_update_view_state") as mock_update:
            with patch.object(page, "refresh_datasets") as mock_refresh:
                page.showEvent(None)
                mock_update.assert_called_once()
                mock_refresh.assert_called_once()

    def test_extra_edge_cases_page(self, tmp_path: Path) -> None:
        """Verify fallback behaviors for missing orchestrators, corrupt metadata, and empty paths."""
        app_obj = DeployAIApplication()
        
        # 1. No orchestrator at all
        page_no_orch = DatasetManagerPage(orchestrator=None)
        assert page_no_orch.get_active_workspace_path() is None
        
        # 2. refresh_datasets when active path is None
        page_no_orch.refresh_datasets() # No crash, returns early
        
        # 3. refresh_datasets when datasets folder doesn't exist
        mock_orch = MagicMock()
        mock_orch.active_workspace_path = str(tmp_path / "missing_dir")
        page_missing_dir = DatasetManagerPage(orchestrator=mock_orch)
        page_missing_dir.refresh_datasets() # No crash, returns early

        # 4. _on_import_dataset when active path is None
        mock_orch_none = MagicMock()
        mock_orch_none.active_workspace_path = None
        page_none = DatasetManagerPage(orchestrator=mock_orch_none)
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page_none._on_import_dataset()
            mock_crit.assert_called_with(page_none, "Import Error", "No active workspace is open.")

        # 5. refresh_datasets skip corrupt JSON files
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        # Save a corrupt json file
        with open(datasets_dir / "bad.json", "w") as f:
            f.write("corrupt { json")
        mock_orch.active_workspace_path = str(tmp_path)
        page_corrupt = DatasetManagerPage(orchestrator=mock_orch)
        page_corrupt.refresh_datasets() # Should not raise error and row count should be 0
        assert page_corrupt.table.rowCount() == 0

        # 6. _on_import_dataset when metadata read fails after Accepted
        meta_file = datasets_dir / "imported.csv.json"
        with open(meta_file, "w") as f:
            f.write("corrupt")
        with patch("frontend.pages.dataset_manager.ImportDatasetDialog") as MockDialog:
            dialog_instance = MockDialog.return_value
            dialog_instance.exec.return_value = QDialog.DialogCode.Accepted
            dialog_instance.imported_metadata_path = str(meta_file.resolve())
            page_corrupt._on_import_dataset() # No crash, handles exception

        # 7. Selection changed with no items
        page_corrupt.table.setColumnCount(1)
        page_corrupt.table.setRowCount(1)
        # Select empty row
        page_corrupt.table.selectRow(0)
        page_corrupt._on_selection_changed()
        assert page_corrupt.details_stack.currentIndex() == 0

        # 8. Selection changed with item but no metadata user role data
        item_empty = QTableWidgetItem("Empty")
        page_corrupt.table.setItem(0, 0, item_empty)
        page_corrupt.table.selectRow(0)
        page_corrupt._on_selection_changed()
        assert page_corrupt.details_stack.currentIndex() == 0

    def test_extra_edge_cases_dialog(self, tmp_path: Path) -> None:
        """Verify dialog edge cases: corrupt files, duplicate json parse crash, copy exception, save exception."""
        app_obj = DeployAIApplication()
        
        # 1. Duplicate check with corrupt metadata json in workspace datasets folder
        datasets_dir = tmp_path / "datasets"
        datasets_dir.mkdir(parents=True)
        with open(datasets_dir / "bad.json", "w") as f:
            f.write("bad-json")
            
        dialog = ImportDatasetDialog(workspace_path=str(tmp_path))
        target = tmp_path / "valid.csv"
        target.write_text("a,b\n1,2", encoding="utf-8")
        dialog.txt_file.setText(str(target))
        dialog.txt_name.setText("UniqueName")
        dialog._on_import() # Passes duplicate check despite bad JSON file

        # 2. Excel empty headers validation
        dialog2 = ImportDatasetDialog(workspace_path=str(tmp_path))
        empty_xlsx = tmp_path / "empty_header.xlsx"
        empty_xlsx.touch()
        dialog2.txt_file.setText(str(empty_xlsx))
        dialog2.txt_name.setText("ExcelEmptyHeaders")
        # Empty header dataframe
        with patch("pandas.read_excel", return_value=pd.DataFrame()):
            with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
                dialog2._on_import()
                mock_warn.assert_called_with(dialog2, "Validation Error", "The dataset file is empty (contains no rows or columns).")

        # 3. Header reading exception
        dialog3 = ImportDatasetDialog(workspace_path=str(tmp_path))
        error_xlsx = tmp_path / "error_header.xlsx"
        error_xlsx.touch()
        dialog3.txt_file.setText(str(error_xlsx))
        dialog3.txt_name.setText("ErrorHeader")
        with patch("pandas.read_excel", side_effect=RuntimeError("xls read error")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                dialog3._on_import()
                mock_crit.assert_called_with(dialog3, "Import Error", "Failed to read file headers: xls read error")

        # 4. shutil.copy2 exception
        dialog4 = ImportDatasetDialog(workspace_path=str(tmp_path))
        target_f = tmp_path / "valid2.csv"
        target_f.write_text("a,b\n1,2", encoding="utf-8")
        dialog4.txt_file.setText(str(target_f))
        dialog4.txt_name.setText("CopyErrorName")
        with patch("shutil.copy2", side_effect=OSError("copy failed")):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                dialog4._on_import()
                mock_crit.assert_called_with(dialog4, "Import Error", "Failed to copy file to workspace: copy failed")

        # 5. Metadata JSON save exception
        dialog5 = ImportDatasetDialog(workspace_path=str(tmp_path))
        dialog5.txt_file.setText(str(target_f))
        dialog5.txt_name.setText("SaveErrorName")
        
        real_open = open
        def mock_open_fn(file, *args, **kwargs):
            if str(file).endswith(".json"):
                raise OSError("save failed")
            return real_open(file, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open_fn):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                dialog5._on_import()
                mock_crit.assert_called_with(dialog5, "Import Error", "Failed to save dataset metadata: save failed")




