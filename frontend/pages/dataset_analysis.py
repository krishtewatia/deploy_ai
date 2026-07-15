"""Dataset Analysis Dashboard widget for DeployAI."""

import os
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QGroupBox,
    QProgressBar,
    QScrollArea,
    QFormLayout,
    QTextEdit,
    QHeaderView,
    QFrame,
)
from PySide6.QtCore import Qt

from backend.app.analysis.analysis_service import AnalysisService  # backend.app.workspace
from backend.app.dataset_intelligence.context_builder import DatasetContextBuilder  # backend.app.workspace
from frontend.pages.dataset_manager import format_size


class DatasetAnalysisWidget(QWidget):
    """The Dataset Analysis Dashboard widget visualising data profiling metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_metadata: dict | None = None

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll Area Wrapper
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        main_layout.addWidget(scroll_area)

        # Content Widget
        content_widget = QWidget(scroll_area)
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)
        scroll_area.setWidget(content_widget)

        # Header Title
        title_label = QLabel("Dataset Analysis", content_widget)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a1a1a;")
        content_layout.addWidget(title_label)

        # ----------------------------------------------------
        # 1. Dataset Summary Section
        # ----------------------------------------------------
        summary_group = QGroupBox("Dataset Summary", content_widget)
        summary_group.setStyleSheet("font-weight: bold; margin-top: 5px;")
        summary_layout = QFormLayout(summary_group)
        summary_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        
        self.lbl_rows = QLabel("-", summary_group)
        self.lbl_cols = QLabel("-", summary_group)
        self.lbl_mem = QLabel("-", summary_group)
        self.lbl_file_size = QLabel("-", summary_group)
        self.lbl_target = QLabel("-", summary_group)
        self.lbl_status = QLabel("-", summary_group)

        # Reset stylesheets for child widgets of GroupBox to default normal weight
        for widget in [self.lbl_rows, self.lbl_cols, self.lbl_mem, self.lbl_file_size, self.lbl_target, self.lbl_status]:
            widget.setStyleSheet("font-weight: normal;")

        summary_layout.addRow("Rows:", self.lbl_rows)
        summary_layout.addRow("Columns:", self.lbl_cols)
        summary_layout.addRow("Memory Usage:", self.lbl_mem)
        summary_layout.addRow("File Size:", self.lbl_file_size)
        summary_layout.addRow("Target Column:", self.lbl_target)
        summary_layout.addRow("Status:", self.lbl_status)
        content_layout.addWidget(summary_group)

        # ----------------------------------------------------
        # 2. Data Quality Section
        # ----------------------------------------------------
        quality_group = QGroupBox("Data Quality", content_widget)
        quality_group.setStyleSheet("font-weight: bold; margin-top: 5px;")
        quality_layout = QFormLayout(quality_group)
        
        self.lbl_missing = QLabel("-", quality_group)
        self.lbl_dup_rows = QLabel("-", quality_group)
        self.lbl_dup_cols = QLabel("-", quality_group)
        self.lbl_const_cols = QLabel("-", quality_group)
        self.lbl_high_card = QLabel("-", quality_group)

        for widget in [self.lbl_missing, self.lbl_dup_rows, self.lbl_dup_cols, self.lbl_const_cols, self.lbl_high_card]:
            widget.setStyleSheet("font-weight: normal;")

        quality_layout.addRow("Missing Values:", self.lbl_missing)
        quality_layout.addRow("Duplicate Rows:", self.lbl_dup_rows)
        quality_layout.addRow("Duplicate Columns:", self.lbl_dup_cols)
        quality_layout.addRow("Constant Columns:", self.lbl_const_cols)
        quality_layout.addRow("High Cardinality Columns:", self.lbl_high_card)
        content_layout.addWidget(quality_group)

        # ----------------------------------------------------
        # 3. Column Overview Section (Table)
        # ----------------------------------------------------
        column_group = QGroupBox("Column Overview", content_widget)
        column_group.setStyleSheet("font-weight: bold; margin-top: 5px;")
        column_layout = QVBoxLayout(column_group)
        column_layout.setContentsMargins(8, 12, 8, 8)

        self.table = QTableWidget(column_group)
        self.table.setStyleSheet("font-weight: normal;")
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Name", "Type", "Missing", "Unique", "Example Value"
        ])
        
        table_header = self.table.horizontalHeader()
        table_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(180)
        column_layout.addWidget(self.table)
        content_layout.addWidget(column_group)

        # ----------------------------------------------------
        # 4. Dataset Readiness Section
        # ----------------------------------------------------
        readiness_group = QGroupBox("Dataset Readiness", content_widget)
        readiness_group.setStyleSheet("font-weight: bold; margin-top: 5px;")
        readiness_layout = QVBoxLayout(readiness_group)
        readiness_layout.setContentsMargins(8, 12, 8, 8)
        readiness_layout.setSpacing(10)

        # Readiness Level and Score Layout
        score_layout = QHBoxLayout()
        score_title = QLabel("Overall Readiness Score:", readiness_group)
        score_title.setStyleSheet("font-weight: bold;")
        self.lbl_readiness_score = QLabel("0%", readiness_group)
        self.lbl_readiness_score.setStyleSheet("font-weight: bold; font-size: 14px;")
        score_layout.addWidget(score_title)
        score_layout.addWidget(self.lbl_readiness_score)
        score_layout.addStretch()
        
        self.lbl_readiness_level = QLabel("-", readiness_group)
        self.lbl_readiness_level.setStyleSheet("font-weight: bold; font-size: 13px;")
        score_layout.addWidget(self.lbl_readiness_level)
        readiness_layout.addLayout(score_layout)

        # Progress Bar visualizing readiness score
        self.progress_bar = QProgressBar(readiness_group)
        self.progress_bar.setStyleSheet("font-weight: normal;")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        readiness_layout.addWidget(self.progress_bar)

        # Warnings Text Box
        warn_lbl = QLabel("Warnings:", readiness_group)
        warn_lbl.setStyleSheet("font-weight: bold;")
        readiness_layout.addWidget(warn_lbl)
        
        self.txt_warnings = QTextEdit(readiness_group)
        self.txt_warnings.setStyleSheet("font-weight: normal; background-color: #fafafa;")
        self.txt_warnings.setReadOnly(True)
        self.txt_warnings.setMaximumHeight(80)
        readiness_layout.addWidget(self.txt_warnings)

        # Recommendations Text Box
        rec_lbl = QLabel("Recommendations:", readiness_group)
        rec_lbl.setStyleSheet("font-weight: bold;")
        readiness_layout.addWidget(rec_lbl)

        self.txt_recommendations = QTextEdit(readiness_group)
        self.txt_recommendations.setStyleSheet("font-weight: normal; background-color: #fafafa;")
        self.txt_recommendations.setReadOnly(True)
        self.txt_recommendations.setMaximumHeight(80)
        readiness_layout.addWidget(self.txt_recommendations)

        content_layout.addWidget(readiness_group)

        # ----------------------------------------------------
        # 5. Bottom Controls (Refresh)
        # ----------------------------------------------------
        self.btn_refresh = QPushButton("Refresh Analysis", content_widget)
        self.btn_refresh.clicked.connect(self._on_refresh)
        self.btn_refresh.setStyleSheet("padding: 6px 12px; font-weight: bold;")
        content_layout.addWidget(self.btn_refresh)

    def set_dataset(self, metadata: dict) -> None:
        """Verify existence on disk, load the dataset, run context builder, and populate layout."""
        self.current_metadata = metadata
        location = metadata.get("location")

        # Validation: Verify file exists on disk
        if not location or not os.path.exists(location):
            QMessageBox.warning(self, "Dataset Error", "The selected dataset file does not exist on disk.")
            self._reset_labels()
            self._trigger_parent_refresh()
            return

        # Validation: Verify metadata is not corrupted
        if not metadata.get("name") or "file_name" not in metadata:
            QMessageBox.warning(self, "Metadata Error", "The dataset metadata is corrupted or incomplete.")
            self._reset_labels()
            return

        try:
            # Load dataset matching file type
            if location.endswith(".csv"):
                df = pd.read_csv(location)
            else:
                df = pd.read_excel(location)

            # Coordinate with analysis services
            analysis_service = AnalysisService()
            report = analysis_service.analyze(df)

            # Build DatasetContext profiles
            builder = DatasetContextBuilder()
            context = builder.build(
                dataset_id=metadata.get("name", "Unknown"),
                file_name=os.path.basename(location),
                row_count=df.shape[0],
                column_count=df.shape[1],
                memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
                analysis_report=report,
            )

            # Populate UI elements
            self._populate_ui_from_context_and_df(context, df)

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Failed to perform dataset analysis: {e}")
            self._reset_labels()

    def _reset_labels(self) -> None:
        """Reset all summary labels to empty indicator values."""
        for label in [self.lbl_rows, self.lbl_cols, self.lbl_mem, self.lbl_file_size, self.lbl_target, self.lbl_status]:
            label.setText("-")
        for label in [self.lbl_missing, self.lbl_dup_rows, self.lbl_dup_cols, self.lbl_const_cols, self.lbl_high_card]:
            label.setText("-")
        self.table.setRowCount(0)
        self.lbl_readiness_score.setText("0%")
        self.lbl_readiness_level.setText("-")
        self.lbl_readiness_level.setStyleSheet("font-weight: bold; font-size: 13px; color: black;")
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("font-weight: normal;")
        self.txt_warnings.clear()
        self.txt_recommendations.clear()

    def _trigger_parent_refresh(self) -> None:
        """Instruct the parent page to clear selections and refresh listings."""
        if self.parent() and self.parent().parent():
            manager = self.parent().parent()
            if hasattr(manager, "refresh_datasets"):
                manager.table.clearSelection()
                manager.refresh_datasets()

    def _populate_ui_from_context_and_df(self, context, df: pd.DataFrame) -> None:
        """Translate structural data context and df to dashboard labels, tables, and bars."""
        # 1. Populate summary fields
        self.lbl_rows.setText(str(context.basic_info.row_count))
        self.lbl_cols.setText(str(context.basic_info.column_count))
        self.lbl_mem.setText(format_size(context.basic_info.memory_usage_bytes))
        
        file_size = self.current_metadata.get("size_bytes", 0) if self.current_metadata else 0
        self.lbl_file_size.setText(format_size(file_size))

        target = self.current_metadata.get("target", "None") if self.current_metadata else "None"
        self.lbl_target.setText(str(target) if target else "None")

        status = self.current_metadata.get("status", "Active") if self.current_metadata else "Active"
        self.lbl_status.setText(status)

        # 2. Compute quality metrics
        missing_cells = context.missing_data.total_missing_cells
        dup_rows = context.duplicates.duplicate_rows
        dup_cols = int(df.T.duplicated().sum())
        const_cols = sum(1 for col in context.columns if col.unique_count == 1)
        high_card = sum(1 for col in context.columns if col.is_categorical and col.unique_count > 20)

        self.lbl_missing.setText(str(missing_cells))
        self.lbl_dup_rows.setText(f"{dup_rows} ({context.duplicates.duplicate_percentage:.1f}%)")
        self.lbl_dup_cols.setText(str(dup_cols))
        self.lbl_const_cols.setText(str(const_cols))
        self.lbl_high_card.setText(str(high_card))

        # 3. Re-fill column overview table
        self.table.setRowCount(0)
        for idx, col in enumerate(context.columns):
            self.table.insertRow(idx)
            
            # Name
            self.table.setItem(idx, 0, QTableWidgetItem(col.name))
            
            # Type
            self.table.setItem(idx, 1, QTableWidgetItem(col.dtype))
            
            # Missing Count & Percentage
            missing_text = f"{col.missing_count} ({col.missing_percentage:.1f}%)"
            self.table.setItem(idx, 2, QTableWidgetItem(missing_text))
            
            # Unique Count & Percentage
            unique_text = f"{col.unique_count} ({col.unique_percentage:.1f}%)"
            self.table.setItem(idx, 3, QTableWidgetItem(unique_text))
            
            # Example value
            example_val = str(col.sample_values[0]) if col.sample_values else "N/A"
            self.table.setItem(idx, 4, QTableWidgetItem(example_val))

            # Make read-only
            for col_idx in range(5):
                item = self.table.item(idx, col_idx)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        # 4. Readiness score algorithm and level
        score = 100
        # Deduct missing cells percentage
        total_cells = context.basic_info.row_count * context.basic_info.column_count
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        score -= min(missing_pct * 2.0, 30.0)

        # Deduct duplicate rows percentage
        score -= min(context.duplicates.duplicate_percentage * 1.5, 20.0)

        # Deduct constant columns (5 points each, max 15)
        score -= min(const_cols * 5.0, 15.0)

        # Deduct duplicate columns (5 points each, max 15)
        score -= min(dup_cols * 5.0, 15.0)

        # Deduct high cardinality columns (3 points each, max 10)
        score -= min(high_card * 3.0, 10.0)

        score = max(0, min(int(round(score)), 100))

        # Set level based on score threshold rules
        if score >= 80:
            level = "Excellent"
            level_color = "#2ec4b6"  # Green
        elif score >= 50:
            level = "Good"
            level_color = "#ff9f1c"  # Yellow
        else:
            level = "Poor"
            level_color = "#e71d36"  # Red

        self.lbl_readiness_score.setText(f"{score}%")
        self.lbl_readiness_level.setText(level)
        self.lbl_readiness_level.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {level_color};")
        
        self.progress_bar.setValue(score)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #d3d3d3;
                border-radius: 4px;
                text-align: center;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {level_color};
                border-radius: 3px;
            }}
        """)

        # 5. Generate Warnings & Recommendations
        warnings = []
        recommendations = []

        if missing_cells > 0:
            missing_cols = [c.name for c in context.columns if c.missing_count > 0]
            warnings.append(f"Missing Values: Dataset has {missing_cells} missing cells across columns: {', '.join(missing_cols)}.")
            recommendations.append(f"Impute missing values or remove records containing nulls in columns: {', '.join(missing_cols)}.")

        if context.duplicates.duplicate_percentage > 5.0:
            warnings.append(f"Duplicate Rows: {context.duplicates.duplicate_percentage:.1f}% duplicate rows detected.")
            recommendations.append("Deduplicate dataset rows using standard drop_duplicates to improve generalization.")

        if dup_cols > 0:
            warnings.append(f"Duplicate Columns: {dup_cols} identical column variables found.")
            recommendations.append("Prune redundant/duplicate column variables to avoid collinearity.")

        if const_cols > 0:
            const_col_names = [c.name for c in context.columns if c.unique_count == 1]
            warnings.append(f"Constant Columns: Found {const_cols} column(s) with single-value variance: {', '.join(const_col_names)}.")
            recommendations.append(f"Exclude constant columns from input schema as they carry zero variance: {', '.join(const_col_names)}.")

        if high_card > 0:
            high_card_names = [c.name for c in context.columns if c.is_categorical and c.unique_count > 20]
            warnings.append(f"High Cardinality: {high_card} categorical column(s) exceed 20 unique values: {', '.join(high_card_names)}.")
            recommendations.append(f"Group low-frequency keys or apply target encoding for features: {', '.join(high_card_names)}.")

        if context.basic_info.row_count < 100:
            warnings.append(f"Small Dataset: Row size is very limited ({context.basic_info.row_count} records).")
            recommendations.append("Collect more data observations if possible to avoid model overfitting.")

        if not warnings:
            self.txt_warnings.setPlainText("No significant warnings detected. The dataset appears clean.")
            self.txt_recommendations.setPlainText("The dataset is ready for planning and modeling stages.")
        else:
            self.txt_warnings.setPlainText("\n".join(f"• {w}" for w in warnings))
            self.txt_recommendations.setPlainText("\n".join(f"• {r}" for r in recommendations))

    def _on_refresh(self) -> None:
        """Run Dataset Intelligence again, update persistent metadata JSON, and refresh UI layout."""
        if not self.current_metadata:
            QMessageBox.warning(self, "Refresh Error", "No dataset is currently selected.")
            return

        location = self.current_metadata.get("location")
        if not location or not os.path.exists(location):
            QMessageBox.warning(self, "Dataset Error", "The selected dataset file does not exist on disk.")
            self._reset_labels()
            self._trigger_parent_refresh()
            return

        try:
            # Reload dataset
            if location.endswith(".csv"):
                df = pd.read_csv(location)
            else:
                df = pd.read_excel(location)

            # Analyze again
            analysis_service = AnalysisService()
            report = analysis_service.analyze(df)

            builder = DatasetContextBuilder()
            context = builder.build(
                dataset_id=self.current_metadata.get("name", "Unknown"),
                file_name=os.path.basename(location),
                row_count=df.shape[0],
                column_count=df.shape[1],
                memory_usage_bytes=int(df.memory_usage(deep=True).sum()),
                analysis_report=report,
            )

            # Re-compute metadata values
            missing_count = int(df.isna().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            size_bytes = os.path.getsize(location)

            # Update cached metadata record
            self.current_metadata["rows"] = df.shape[0]
            self.current_metadata["columns"] = df.shape[1]
            self.current_metadata["size_bytes"] = size_bytes
            self.current_metadata["missing_values"] = missing_count
            self.current_metadata["duplicate_rows"] = duplicate_rows
            self.current_metadata["column_names"] = df.columns.tolist()

            # Save changes to metadata JSON configuration file (does not mutate dataset file)
            meta_path = Path(location).with_suffix(Path(location).suffix + ".json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.current_metadata, f, indent=4)

            # Re-draw UI
            self._populate_ui_from_context_and_df(context, df)

            # Refresh table in DatasetManagerPage
            if self.parent() and self.parent().parent():
                manager = self.parent().parent()
                if hasattr(manager, "refresh_datasets"):
                    sel_name = self.current_metadata.get("name", "")
                    manager.refresh_datasets()
                    manager.select_dataset_by_name(sel_name)

            QMessageBox.information(self, "Success", "Dataset analysis refreshed successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Refresh Error", f"Failed to refresh dataset analysis: {e}")
