"""Dataset Manager page for DeployAI managing dataset uploads and metadata."""

import os
import json
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QStackedWidget,
    QFormLayout,
    QHeaderView,
    QDialog,
)
from PySide6.QtCore import Qt
from frontend.pages.dialogs.import_dataset_dialog import ImportDatasetDialog


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp to human-readable format."""
    try:
        # Handle trailing Z representing UTC
        normalized = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return iso_str


class DatasetManagerPage(QWidget):
    """The Dataset Management page linking to the Dataset Intelligence subsystem."""

    def __init__(self, orchestrator=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator

        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # ----------------------------------------------------
        # Left Panel (Actions and Table of Imported Datasets)
        # ----------------------------------------------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Title
        title_label = QLabel("Datasets", left_widget)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(title_label)

        # Action Buttons Layout
        actions_layout = QHBoxLayout()
        self.btn_import = QPushButton("Import Dataset", left_widget)
        self.btn_import.clicked.connect(self._on_import_dataset)
        
        self.btn_remove = QPushButton("Remove Dataset", left_widget)
        self.btn_remove.setEnabled(False) # Keep disabled as required by specs
        
        actions_layout.addWidget(self.btn_import)
        actions_layout.addWidget(self.btn_remove)
        actions_layout.addStretch()
        left_layout.addLayout(actions_layout)

        # Imported Datasets Table
        self.table = QTableWidget(left_widget)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Name", "File Type", "Rows", "Columns", "Size", "Imported Time"
        ])
        
        # Configure Table Header stretching and resizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.table)

        main_layout.addWidget(left_widget, stretch=6)

        # ----------------------------------------------------
        # Right Panel (Dataset Details)
        # ----------------------------------------------------
        self.details_stack = QStackedWidget(self)
        
        # Details Stack Page 0: Empty/Placeholder
        self.empty_details_label = QLabel("No dataset selected", self.details_stack)
        self.empty_details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_stack.addWidget(self.empty_details_label)

        # Details Stack Page 1: Active Dataset Details
        from frontend.pages.dataset_analysis import DatasetAnalysisWidget
        self.details_widget = DatasetAnalysisWidget(self.details_stack)
        self.details_stack.addWidget(self.details_widget)
        
        main_layout.addWidget(self.details_stack, stretch=4)

        # Initial view state update
        self._update_view_state()

    def showEvent(self, event) -> None:
        """Trigger view state updates and dataset listing when page is shown."""
        super().showEvent(event)
        self._update_view_state()
        self.refresh_datasets()

    def _update_view_state(self) -> None:
        """Enable/Disable controls based on whether a project workspace is active."""
        active_path = self.get_active_workspace_path()
        if active_path:
            self.btn_import.setEnabled(True)
            self.empty_details_label.setText("No dataset selected")
        else:
            self.btn_import.setEnabled(False)
            self.empty_details_label.setText("Please open or create a project workspace first.")
            self.details_stack.setCurrentIndex(0)

    def get_active_workspace_path(self) -> str | None:
        """Retrieve active workspace path from orchestrator instance."""
        if not self.orchestrator:
            return None
        return getattr(self.orchestrator, "active_workspace_path", None)

    def refresh_datasets(self) -> None:
        """Scan active workspace's datasets folder for metadata JSON files and rebuild the table."""
        self.table.setRowCount(0)
        active_path = self.get_active_workspace_path()
        if not active_path:
            return

        datasets_dir = Path(active_path) / "datasets"
        if not datasets_dir.exists():
            return

        row = 0
        for meta_file in sorted(datasets_dir.glob("*.json"), key=os.path.getmtime):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                self.table.insertRow(row)

                # Set items
                item_name = QTableWidgetItem(metadata.get("name", ""))
                # Store full metadata inside the name item for selection retrieve
                item_name.setData(Qt.ItemDataRole.UserRole, metadata)

                item_type = QTableWidgetItem(metadata.get("file_type", ""))
                item_rows = QTableWidgetItem(str(metadata.get("rows", 0)))
                item_cols = QTableWidgetItem(str(metadata.get("columns", 0)))
                item_size = QTableWidgetItem(format_size(metadata.get("size_bytes", 0)))
                item_time = QTableWidgetItem(format_timestamp(metadata.get("imported_time", "")))

                # Make table non-editable
                for col_idx, item in enumerate([item_name, item_type, item_rows, item_cols, item_size, item_time]):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, col_idx, item)

                row += 1
            except Exception:
                pass

    def select_dataset_by_name(self, name: str) -> None:
        """Find and select dataset row matching the given name."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text().lower() == name.lower():
                self.table.selectRow(row)
                break

    def _on_import_dataset(self) -> None:
        """Open the ImportDatasetDialog."""
        active_path = self.get_active_workspace_path()
        if not active_path:
            QMessageBox.critical(self, "Import Error", "No active workspace is open.")
            return

        dialog = ImportDatasetDialog(active_path, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.refresh_datasets()
            
            # Select the newly imported dataset automatically
            if dialog.imported_metadata_path:
                try:
                    with open(dialog.imported_metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        self.select_dataset_by_name(meta.get("name", ""))
                except Exception:
                    pass

    def _on_selection_changed(self) -> None:
        """Update the details panel when a row is selected."""
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            self.details_stack.setCurrentIndex(0)
            return

        row = selected_ranges[0].topRow()
        item = self.table.item(row, 0)
        if not item:
            self.details_stack.setCurrentIndex(0)
            return

        metadata = item.data(Qt.ItemDataRole.UserRole)
        if not metadata:
            self.details_stack.setCurrentIndex(0)
            return

        self._populate_details(metadata)
        self.details_stack.setCurrentIndex(1)

    def _populate_details(self, metadata: dict) -> None:
        """Fill details with dataset analysis dashboard."""
        self.details_widget.set_dataset(metadata)
