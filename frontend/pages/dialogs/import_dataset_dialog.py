"""Import Dataset Dialog for DeployAI page workspace integration."""

import os
import shutil
import json
from datetime import datetime, timezone
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QWidget,
)
from PySide6.QtCore import Qt
import pandas as pd


class ImportDatasetDialog(QDialog):
    """Modal dialog to select, validate, and import a dataset into the project workspace."""

    def __init__(self, workspace_path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workspace_path = workspace_path
        self.setWindowTitle("Import Dataset")
        self.resize(500, 200)

        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Form Layout
        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        # File selection field
        file_layout = QHBoxLayout()
        self.txt_file = QLineEdit(self)
        self.txt_file.setPlaceholderText("Select CSV or Excel file...")
        self.btn_browse = QPushButton("Browse...", self)
        self.btn_browse.clicked.connect(self._on_browse)
        file_layout.addWidget(self.txt_file)
        file_layout.addWidget(self.btn_browse)

        # Name field
        self.txt_name = QLineEdit(self)
        self.txt_name.setPlaceholderText("Enter a unique dataset name...")

        form_layout.addRow("Dataset File:", file_layout)
        form_layout.addRow("Dataset Name:", self.txt_name)
        layout.addLayout(form_layout)

        # Dialog Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_import = QPushButton("Import", self)
        self.btn_import.clicked.connect(self._on_import)
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_import)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        # Result storage
        self.imported_metadata_path: str | None = None

    def _on_browse(self) -> None:
        """Open QFileDialog to pick a dataset file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset File",
            "",
            "Dataset Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )
        if file_path:
            self.txt_file.setText(file_path)
            # If name is empty, auto-populate with file name without extension
            if not self.txt_name.text().strip():
                base_name = Path(file_path).stem
                self.txt_name.setText(base_name)

    def _on_import(self) -> None:
        """Validate, load, copy, and save metadata of the dataset."""
        file_path = self.txt_file.text().strip()
        dataset_name = self.txt_name.text().strip()

        # 1. Validation: Workspace Open
        if not self.workspace_path or not os.path.isdir(self.workspace_path):
            QMessageBox.critical(self, "Import Error", "No active workspace is open.")
            return

        # 2. Validation: Empty inputs
        if not file_path:
            QMessageBox.warning(self, "Validation Error", "Please select a dataset file.")
            return
        if not dataset_name:
            QMessageBox.warning(self, "Validation Error", "Please enter a dataset name.")
            return

        # 3. Validation: File Existence
        src_path = Path(file_path)
        if not src_path.is_file():
            QMessageBox.warning(self, "Validation Error", "Selected file does not exist.")
            return

        # 4. Validation: File Type / Extension
        ext = src_path.suffix.lower()
        if ext not in {".csv", ".xlsx", ".xls"}:
            QMessageBox.warning(self, "Validation Error", "Unsupported file extension. Only CSV and Excel are supported.")
            return

        # 5. Validation: Duplicate name in workspace
        datasets_dir = Path(self.workspace_path) / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        for meta_file in datasets_dir.glob("*.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("name", "").strip().lower() == dataset_name.lower():
                        QMessageBox.warning(self, "Validation Error", f"A dataset named '{dataset_name}' already exists in this project.")
                        return
            except Exception:
                pass

        # 6. Read and validate structure with Pandas
        import csv
        try:
            if ext == ".csv":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    headers = next(reader)
            else:
                # Excel: read first row
                df_header = pd.read_excel(file_path, header=None, nrows=1)
                if df_header.empty:
                    headers = []
                else:
                    headers = [str(x) for x in df_header.iloc[0].tolist()]
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read file headers: {e}")
            return

        # Duplicate columns check from raw headers
        seen = set()
        duplicates = []
        for h in headers:
            h_str = str(h).strip()
            if h_str in seen:
                if h_str not in duplicates:
                    duplicates.append(h_str)
            seen.add(h_str)
        if duplicates:
            QMessageBox.warning(self, "Validation Error", f"Duplicate column names are not allowed: {duplicates}")
            return

        try:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to read file: {e}")
            return

        # Empty check
        if df.shape[0] == 0 or df.shape[1] == 0:
            QMessageBox.warning(self, "Validation Error", "The dataset file is empty (contains no rows or columns).")
            return


        # 7. Generate a collision-safe destination filename
        orig_name = src_path.name
        dest_file = datasets_dir / orig_name
        counter = 1
        while dest_file.exists():
            stem = src_path.stem
            dest_file = datasets_dir / f"{stem}_{counter}{ext}"
            counter += 1

        # 8. Copy file
        try:
            shutil.copy2(src_path, dest_file)
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to copy file to workspace: {e}")
            return

        # 9. Compute stats and create metadata
        try:
            missing_count = int(df.isna().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            size_bytes = os.path.getsize(dest_file)
            imported_time = datetime.now(timezone.utc).isoformat()
            
            metadata = {
                "name": dataset_name,
                "file_name": dest_file.name,
                "file_type": "CSV" if ext == ".csv" else "Excel",
                "rows": df.shape[0],
                "columns": df.shape[1],
                "size_bytes": size_bytes,
                "imported_time": imported_time,
                "location": str(dest_file.resolve()),
                "missing_values": missing_count,
                "duplicate_rows": duplicate_rows,
                "target": "None",
                "status": "Active",
                "column_names": df.columns.tolist(),
            }

            meta_path = dest_file.with_suffix(dest_file.suffix + ".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            
            self.imported_metadata_path = str(meta_path.resolve())
            self.accept()
        except Exception as e:
            # Clean up copied file on failure
            dest_file.unlink(missing_ok=True)
            QMessageBox.critical(self, "Import Error", f"Failed to save dataset metadata: {e}")
