"""Project Dashboard page for DeployAI managing project workspaces."""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QFileDialog,
    QStackedWidget,
    QFormLayout,
    QMenu,
    QDialog,
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QAction
from frontend.pages.dialogs.new_project_dialog import NewProjectDialog
from backend.app.workspace import WorkspaceOrchestrator, ProjectWorkspace
from backend.app.workspace.orchestrator import WorkspaceOrchestratorError


class ProjectDashboardPage(QWidget):
    """The Project Dashboard page linking to the Workspace Orchestrator backend."""

    def __init__(self, orchestrator: WorkspaceOrchestrator | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.orchestrator = orchestrator

        # Main horizontal layout: left navigation and list, right details panel
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)

        # ----------------------------------------------------
        # Left Panel (Actions and Recent Projects List)
        # ----------------------------------------------------
        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Title
        title_label = QLabel("DeployAI Projects", left_widget)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(title_label)

        # Action Buttons Layout
        actions_layout = QHBoxLayout()
        self.new_btn = QPushButton("New Project", left_widget)
        self.new_btn.clicked.connect(self._on_new_project)
        self.open_btn = QPushButton("Open Existing Project", left_widget)
        self.open_btn.clicked.connect(self._on_open_project)
        actions_layout.addWidget(self.new_btn)
        actions_layout.addWidget(self.open_btn)
        left_layout.addLayout(actions_layout)

        # Recent Projects Label
        recent_label = QLabel("Recent Projects", left_widget)
        recent_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        left_layout.addWidget(recent_label)

        # Recent Projects List
        self.recent_list = QListWidget(left_widget)
        self.recent_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.recent_list.customContextMenuRequested.connect(self._show_context_menu)
        self.recent_list.itemSelectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.recent_list)

        main_layout.addWidget(left_widget, stretch=4)

        # ----------------------------------------------------
        # Right Panel (Project Details)
        # ----------------------------------------------------
        self.details_stack = QStackedWidget(self)
        
        # Details Stack Page 0: Empty/Placeholder
        self.empty_details_label = QLabel("No project selected", self.details_stack)
        self.empty_details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.details_stack.addWidget(self.empty_details_label)

        # Details Stack Page 1: Active Project Details
        self.details_widget = QWidget(self.details_stack)
        details_layout = QVBoxLayout(self.details_widget)
        details_layout.setContentsMargins(10, 10, 10, 10)
        details_layout.setSpacing(15)

        details_title = QLabel("Project Details", self.details_widget)
        details_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        details_layout.addWidget(details_title)

        # Form layout for detailed fields
        self.form_layout = QFormLayout()
        
        self.lbl_name = QLabel(self.details_widget)
        self.lbl_type = QLabel(self.details_widget)
        self.lbl_status = QLabel(self.details_widget)
        self.lbl_created = QLabel(self.details_widget)
        self.lbl_path = QLabel(self.details_widget)
        self.lbl_tags = QLabel(self.details_widget)
        
        self.lbl_description = QLabel(self.details_widget)
        self.lbl_description.setWordWrap(True)

        self.form_layout.addRow("Project Name:", self.lbl_name)
        self.form_layout.addRow("Project Type:", self.lbl_type)
        self.form_layout.addRow("Status:", self.lbl_status)
        self.form_layout.addRow("Created Date:", self.lbl_created)
        self.form_layout.addRow("Workspace Path:", self.lbl_path)
        self.form_layout.addRow("Tags:", self.lbl_tags)
        self.form_layout.addRow("Description:", self.lbl_description)

        details_layout.addLayout(self.form_layout)
        details_layout.addStretch()

        self.details_stack.addWidget(self.details_widget)
        
        main_layout.addWidget(self.details_stack, stretch=6)

        # Initial refresh of the list
        self.refresh_recent_projects()

    def refresh_recent_projects(self) -> None:
        """Fetch recent projects from the orchestrator and rebuild list widget."""
        self.recent_list.clear()
        if not self.orchestrator:
            return

        try:
            entries = self.orchestrator.list_recent_projects()
            for entry in entries:
                pin_text = " [Pinned]" if entry.pinned else ""
                item_title = f"{entry.project_name}{pin_text}"
                item = QListWidgetItem(f"{item_title}\n{entry.workspace_path}", self.recent_list)
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self.recent_list.addItem(item)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to list recent projects: {e}")

    def select_project_by_path(self, path: str) -> None:
        """Find and select project in the list matching the given path."""
        for idx in range(self.recent_list.count()):
            item = self.recent_list.item(idx)
            entry = item.data(Qt.ItemDataRole.UserRole)
            if entry and entry.workspace_path == path:
                self.recent_list.setCurrentItem(item)
                break

    # ----------------------------------------------------
    # Event Handlers
    # ----------------------------------------------------

    def _on_new_project(self) -> None:
        """Open the NewProjectDialog to create a project."""
        if not self.orchestrator:
            QMessageBox.critical(self, "Backend Error", "Workspace Orchestrator not initialized.")
            return

        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            metadata = dialog.get_metadata()
            folder = dialog.get_workspace_folder()
            try:
                self.orchestrator.create_project(folder, metadata)
                self.refresh_recent_projects()
                self.select_project_by_path(folder)
            except WorkspaceOrchestratorError as e:
                QMessageBox.critical(self, "Creation Error", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Unexpected Error", f"Failed to create project: {e}")

    def _on_open_project(self) -> None:
        """Select existing folder and open it as a project."""
        if not self.orchestrator:
            QMessageBox.critical(self, "Backend Error", "Workspace Orchestrator not initialized.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Open Existing Project Workspace")
        if folder:
            try:
                self.orchestrator.open_project(folder)
                self.refresh_recent_projects()
                self.select_project_by_path(folder)
            except WorkspaceOrchestratorError as e:
                QMessageBox.critical(self, "Open Error", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Unexpected Error", f"Failed to open project: {e}")

    def _on_selection_changed(self) -> None:
        """Update the details panel when list selection changes."""
        item = self.recent_list.currentItem()
        if not item or not self.orchestrator:
            self.details_stack.setCurrentIndex(0)
            if self.orchestrator:
                self.orchestrator.active_workspace_path = None
                self.orchestrator.active_workspace = None
            return

        entry = item.data(Qt.ItemDataRole.UserRole)
        if not entry:
            self.details_stack.setCurrentIndex(0)
            if self.orchestrator:
                self.orchestrator.active_workspace_path = None
                self.orchestrator.active_workspace = None
            return

        try:
            ws = self.orchestrator.open_project(entry.workspace_path)
            self._populate_details(ws)
            self.details_stack.setCurrentIndex(1)
            self.orchestrator.active_workspace_path = entry.workspace_path
            self.orchestrator.active_workspace = ws
        except Exception:
            self.details_stack.setCurrentIndex(0)
            self.orchestrator.active_workspace_path = None
            self.orchestrator.active_workspace = None


    def _populate_details(self, ws: ProjectWorkspace) -> None:
        """Fill detail labels with ProjectWorkspace metadata."""
        meta = ws.metadata
        self.lbl_name.setText(meta.project_name)
        self.lbl_type.setText(meta.project_type)
        self.lbl_status.setText(ws.status)
        self.lbl_created.setText(meta.created_timestamp)
        self.lbl_path.setText(ws.root_path)
        self.lbl_tags.setText(", ".join(meta.tags) if meta.tags else "None")
        self.lbl_description.setText(meta.description if meta.description else "No description.")

    def _show_context_menu(self, pos: QPoint) -> None:
        """Display the context menu for Pin, Unpin, and Delete options."""
        item = self.recent_list.itemAt(pos)
        if not item or not self.orchestrator:
            return

        entry = item.data(Qt.ItemDataRole.UserRole)
        if not entry:
            return

        menu = QMenu(self)
        
        # Pin/Unpin actions
        if entry.pinned:
            unpin_action = QAction("Unpin", self)
            unpin_action.triggered.connect(lambda: self._unpin_project(entry.workspace_path))
            menu.addAction(unpin_action)
        else:
            pin_action = QAction("Pin", self)
            pin_action.triggered.connect(lambda: self._pin_project(entry.workspace_path))
            menu.addAction(pin_action)

        # Delete placeholder
        delete_action = QAction("Delete (Not Implemented)", self)
        delete_action.setEnabled(False)
        menu.addAction(delete_action)

        menu.exec(self.recent_list.mapToGlobal(pos))

    def _pin_project(self, path: str) -> None:
        """Delegate pinning to the orchestrator."""
        try:
            self.orchestrator.pin_project(path)
            self.refresh_recent_projects()
            self.select_project_by_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Pin Error", f"Failed to pin project: {e}")

    def _unpin_project(self, path: str) -> None:
        """Delegate unpinning to the orchestrator."""
        try:
            self.orchestrator.unpin_project(path)
            self.refresh_recent_projects()
            self.select_project_by_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Unpin Error", f"Failed to unpin project: {e}")
