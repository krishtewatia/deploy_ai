"""New Project Dialog for creating a DeployAI project."""

from datetime import datetime, timezone
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QComboBox,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QWidget,
)
from backend.app.workspace.schemas import ProjectMetadata, ProjectType


class NewProjectDialog(QDialog):
    """Dialog prompting the user for new project details."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(450)

        # Form Layout for fields
        form_layout = QFormLayout()

        self.name_edit = QLineEdit(self)
        form_layout.addRow("Project Name *:", self.name_edit)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems([t.value for t in ProjectType])
        form_layout.addRow("Project Type *:", self.type_combo)

        self.author_edit = QLineEdit(self)
        form_layout.addRow("Author *:", self.author_edit)

        self.tags_edit = QLineEdit(self)
        self.tags_edit.setPlaceholderText("Comma-separated tags (e.g. classifier, tabular)")
        form_layout.addRow("Tags:", self.tags_edit)

        self.description_edit = QTextEdit(self)
        self.description_edit.setAcceptRichText(False)
        form_layout.addRow("Description:", self.description_edit)

        # Workspace Folder selection layout
        folder_layout = QHBoxLayout()
        self.folder_edit = QLineEdit(self)
        self.browse_btn = QPushButton("Browse...", self)
        self.browse_btn.clicked.connect(self._on_browse)
        folder_layout.addWidget(self.folder_edit)
        folder_layout.addWidget(self.browse_btn)
        form_layout.addRow("Workspace Folder *:", folder_layout)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reject)
        
        self.create_btn = QPushButton("Create", self)
        self.create_btn.setDefault(True)
        self.create_btn.clicked.connect(self._on_create)

        btn_layout.addStretch()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.create_btn)

        # Main Layout
        main_layout = QVBoxLayout(self)
        main_layout.addLayout(form_layout)
        main_layout.addLayout(btn_layout)

    def _on_browse(self) -> None:
        """Open QFileDialog to select existing directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Workspace Folder")
        if dir_path:
            self.folder_edit.setText(dir_path)

    def _on_create(self) -> None:
        """Validate input fields and accept dialog if valid."""
        name = self.name_edit.text().strip()
        author = self.author_edit.text().strip()
        folder = self.folder_edit.text().strip()

        # Validation checks
        if not name:
            QMessageBox.warning(self, "Validation Error", "Project Name cannot be empty.")
            return

        if not author:
            QMessageBox.warning(self, "Validation Error", "Author cannot be empty.")
            return

        if not folder:
            QMessageBox.warning(self, "Validation Error", "Workspace Folder cannot be empty.")
            return

        # Check if the folder path string looks invalid (contains invalid character combinations)
        # Detailed path syntax checks are handled by the orchestrator, but we check basic non-empty string here.
        self.accept()

    def get_metadata(self) -> ProjectMetadata:
        """Build and return ProjectMetadata schema from fields."""
        name = self.name_edit.text().strip()
        project_type = ProjectType(self.type_combo.currentText())
        author = self.author_edit.text().strip()
        
        # Tags parser
        tags_raw = self.tags_edit.text().strip()
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []

        # Description: combine Author and Description
        desc_body = self.description_edit.toPlainText().strip()
        description = f"Author: {author}\n\n{desc_body}".strip()

        return ProjectMetadata(
            project_name=name,
            project_type=project_type,
            created_timestamp=datetime.now(timezone.utc).isoformat(),
            description=description,
            tags=tags,
        )

    def get_workspace_folder(self) -> str:
        """Return the selected workspace folder path."""
        return self.folder_edit.text().strip()
