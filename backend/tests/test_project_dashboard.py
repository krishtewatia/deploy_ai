"""Unit and integration tests for the Project Dashboard and New Project Dialog (Stage 12D)."""

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QListWidgetItem
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QAction
from frontend.app import DeployAIApplication
from frontend.pages.project_dashboard import ProjectDashboardPage
from frontend.pages.dialogs.new_project_dialog import NewProjectDialog
from backend.app.workspace import ProjectWorkspace, ProjectStatus, ProjectMetadata, ProjectType
from backend.app.workspace.recent_registry import RecentProjectEntry
from backend.app.workspace.orchestrator import WorkspaceOrchestratorError


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def _make_mock_entry(path: str = "/path/to/p1", name: str = "P1", pinned: bool = False) -> RecentProjectEntry:
    return RecentProjectEntry(
        project_id="ws-123",
        project_name=name,
        workspace_path=path,
        last_opened_timestamp="2025-06-15T12:00:00Z",
        pinned=pinned
    )


def _make_mock_workspace(path: str = "/path/to/p1", name: str = "P1") -> ProjectWorkspace:
    return ProjectWorkspace(
        workspace_id="ws-123",
        status=ProjectStatus.ACTIVE,
        metadata=ProjectMetadata(
            project_name=name,
            project_type=ProjectType.CLASSIFICATION,
            created_timestamp="2025-06-15T12:00:00Z",
            description="Author: TestUser\n\nDesc body",
            tags=["tag1", "tag2"]
        ),
        root_path=path,
        folders=[]
    )


# =======================================================================
# NewProjectDialog Tests
# =======================================================================

class TestNewProjectDialog:

    def test_dialog_fields_initial_state(self) -> None:
        """Verify the dialog initializes fields correctly."""
        # Need a QApplication context
        app_obj = DeployAIApplication()
        dialog = NewProjectDialog(app_obj.main_window)

        assert dialog.name_edit.text() == ""
        assert dialog.author_edit.text() == ""
        assert dialog.folder_edit.text() == ""
        assert dialog.type_combo.count() == len(ProjectType)

    def test_dialog_browse_button(self) -> None:
        """Verify the browse folder button updates the folder field."""
        app_obj = DeployAIApplication()
        dialog = NewProjectDialog(app_obj.main_window)

        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="/selected/folder"):
            dialog._on_browse()
            assert dialog.folder_edit.text() == "/selected/folder"

    def test_dialog_validation_empty_fields(self) -> None:
        """Verify dialog rejects empty name, author, or workspace path."""
        app_obj = DeployAIApplication()
        dialog = NewProjectDialog(app_obj.main_window)

        # 1. Empty name
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_create()
            mock_warn.assert_called_with(dialog, "Validation Error", "Project Name cannot be empty.")

        # 2. Empty author
        dialog.name_edit.setText("My Project")
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_create()
            mock_warn.assert_called_with(dialog, "Validation Error", "Author cannot be empty.")

        # 3. Empty folder
        dialog.author_edit.setText("Jane Doe")
        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warn:
            dialog._on_create()
            mock_warn.assert_called_with(dialog, "Validation Error", "Workspace Folder cannot be empty.")

        # 4. Valid fields
        dialog.folder_edit.setText("/path/to/workspace")
        dialog._on_create()
        # Should call accept() which marks dialog result as accepted
        assert dialog.result() == QDialog.DialogCode.Accepted

    def test_dialog_get_metadata_and_folder(self) -> None:
        """Verify correct ProjectMetadata building and formatting from fields."""
        app_obj = DeployAIApplication()
        dialog = NewProjectDialog(app_obj.main_window)

        dialog.name_edit.setText("My Project")
        dialog.type_combo.setCurrentText("CLASSIFICATION")
        dialog.author_edit.setText("Alice")
        dialog.tags_edit.setText("tabular, classifier")
        dialog.description_edit.setPlainText("Test Description")
        dialog.folder_edit.setText("/my/workspace")

        metadata = dialog.get_metadata()
        assert metadata.project_name == "My Project"
        assert metadata.project_type == ProjectType.CLASSIFICATION
        assert "Author: Alice" in metadata.description
        assert "Test Description" in metadata.description
        assert metadata.tags == ["tabular", "classifier"]

        assert dialog.get_workspace_folder() == "/my/workspace"


# =======================================================================
# ProjectDashboardPage Tests
# =======================================================================

class TestProjectDashboardPage:

    def test_dashboard_initial_load_empty(self) -> None:
        """Verify the dashboard layout displays correctly when no recent projects exist."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = []

        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        assert page.recent_list.count() == 0
        assert page.details_stack.currentIndex() == 0  # Page 0 is placeholder

    def test_dashboard_initial_load_with_projects(self) -> None:
        """Verify the dashboard populates list widget when recent projects exist."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = [
            _make_mock_entry("/path/p1", "P1", pinned=True),
            _make_mock_entry("/path/p2", "P2", pinned=False),
        ]

        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        assert page.recent_list.count() == 2
        
        item1 = page.recent_list.item(0)
        assert "P1 [Pinned]" in item1.text()
        
        item2 = page.recent_list.item(1)
        assert "P2" in item2.text()
        assert "[Pinned]" not in item2.text()

    def test_selection_changed_populates_details(self) -> None:
        """Verify details panel updates when list selection changes."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = [
            _make_mock_entry("/path/p1", "P1")
        ]
        mock_orch.open_project.return_value = _make_mock_workspace("/path/p1", "P1")

        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        # Select first item
        page.recent_list.setCurrentRow(0)
        
        assert page.details_stack.currentIndex() == 1  # Active project details page
        assert page.lbl_name.text() == "P1"
        assert page.lbl_type.text() == "CLASSIFICATION"
        assert page.lbl_status.text() == "ACTIVE"
        assert page.lbl_path.text() == "/path/p1"
        assert page.lbl_tags.text() == "tag1, tag2"
        assert "Author: TestUser" in page.lbl_description.text()

    def test_selection_changed_handles_load_error(self) -> None:
        """Verify details panel falls back to empty page if workspace loading fails."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = [
            _make_mock_entry("/path/p1", "P1")
        ]
        mock_orch.open_project.side_effect = RuntimeError("disk corrupt")

        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        # Select first item
        page.recent_list.setCurrentRow(0)
        assert page.details_stack.currentIndex() == 0  # Reverted to placeholder

    def test_new_project_creation_success(self) -> None:
        """Verify clicking 'New Project' dialog success flow triggers creation and refresh."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.side_effect = [
            [], # Init list call
            [_make_mock_entry("/new/project", "New Project")] # After creation refresh call
        ]
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        # Setup mock dialog behavior
        mock_metadata = ProjectMetadata(
            project_name="New Project",
            project_type=ProjectType.REGRESSION,
            created_timestamp="now",
            description="Author: Alice\n\nDesc"
        )
        
        # Mock open_project to return a valid ProjectWorkspace
        mock_orch.open_project.return_value = _make_mock_workspace("/new/project", "New Project")

        with patch("frontend.pages.project_dashboard.NewProjectDialog") as MockDialog:
            dialog_instance = MockDialog.return_value
            dialog_instance.exec.return_value = QDialog.DialogCode.Accepted
            dialog_instance.get_metadata.return_value = mock_metadata
            dialog_instance.get_workspace_folder.return_value = "/new/project"

            page._on_new_project()
            
            # Verify mock orchestrator create was called with correct data
            mock_orch.create_project.assert_called_with("/new/project", mock_metadata)
            assert page.recent_list.count() == 1

    def test_new_project_creation_error_surfaced(self) -> None:
        """Verify workspace errors are surfaced in a critical messagebox."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = []
        mock_orch.create_project.side_effect = WorkspaceOrchestratorError("injected creation failure")
        mock_orch.open_project.return_value = _make_mock_workspace("/fail/project", "Fail")
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        with patch("frontend.pages.project_dashboard.NewProjectDialog") as MockDialog:
            dialog_instance = MockDialog.return_value
            dialog_instance.exec.return_value = QDialog.DialogCode.Accepted
            dialog_instance.get_metadata.return_value = _make_mock_workspace().metadata
            dialog_instance.get_workspace_folder.return_value = "/fail/project"

            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                page._on_new_project()
                mock_crit.assert_called_with(page, "Creation Error", "injected creation failure")


    def test_open_project_success(self) -> None:
        """Verify open existing project workspace browsing and loading flow."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.side_effect = [
            [],
            [_make_mock_entry("/open/path", "Opened P")]
        ]
        mock_orch.open_project.return_value = _make_mock_workspace("/open/path", "Opened P")
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)

        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="/open/path"):
            page._on_open_project()
            mock_orch.open_project.assert_called_with("/open/path")
            assert page.recent_list.count() == 1


    def test_open_project_error_surfaced(self) -> None:
        """Verify open errors are surfaced in a critical messagebox."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = []
        mock_orch.open_project.side_effect = WorkspaceOrchestratorError("injected open failure")
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)

        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="/fail/path"):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                page._on_open_project()
                mock_crit.assert_called_with(page, "Open Error", "injected open failure")

    def test_pin_and_unpin_context_menu_triggers(self) -> None:
        """Verify right click menu actions trigger pin/unpin on orchestrator."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.side_effect = [
            [_make_mock_entry("/path/p1", "P1", pinned=False)], # 1. Init
            [_make_mock_entry("/path/p1", "P1", pinned=True)],  # 2. After pin
            [_make_mock_entry("/path/p1", "P1", pinned=False)]  # 3. After unpin
        ]
        mock_orch.open_project.return_value = _make_mock_workspace("/path/p1", "P1")
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        # Mock right click context menu trigger dynamically to avoid C++ object deletion after refresh
        pos = QPoint(10, 10)
        with patch.object(page.recent_list, "itemAt", side_effect=lambda pos: page.recent_list.item(0)):
            with patch("frontend.pages.project_dashboard.QMenu") as MockMenu:

                # 1. Trigger context menu (Unpinned → displays 'Pin' action)
                page._show_context_menu(pos)
                MockMenu.return_value.exec.assert_called()
                
                # Retrieve the actions added to the menu
                added_actions = [call.args[0] for call in MockMenu.return_value.addAction.call_args_list]
                pin_action = next(act for act in added_actions if act.text() == "Pin")
                pin_action.trigger() # Trigger lambda
                mock_orch.pin_project.assert_called_with("/path/p1")

                # 2. Re-trigger menu after pinning to test 'Unpin' action
                MockMenu.return_value.addAction.reset_mock()
                page._show_context_menu(pos)
                
                added_actions2 = [call.args[0] for call in MockMenu.return_value.addAction.call_args_list]
                unpin_action = next(act for act in added_actions2 if act.text() == "Unpin")
                unpin_action.trigger() # Trigger lambda
                mock_orch.unpin_project.assert_called_with("/path/p1")

    def test_no_orchestrator_graceful_fallbacks(self) -> None:
        """Verify no errors are thrown if orchestrator is None."""
        app_obj = DeployAIApplication()
        page = ProjectDashboardPage(orchestrator=None, parent=app_obj.main_window)

        # Trigger actions
        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page._on_new_project()
            mock_crit.assert_called_with(page, "Backend Error", "Workspace Orchestrator not initialized.")

        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page._on_open_project()
            mock_crit.assert_called_with(page, "Backend Error", "Workspace Orchestrator not initialized.")

        # Selection changes should just reset details panel index
        page._on_selection_changed()
        assert page.details_stack.currentIndex() == 0

    def test_pin_and_unpin_graceful_error_wrapping(self) -> None:
        """Verify pin/unpin exceptions are cleanly wrapped and reported."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = [
            _make_mock_entry("/path/p1", "P1")
        ]
        mock_orch.pin_project.side_effect = RuntimeError("pin crash")
        mock_orch.unpin_project.side_effect = RuntimeError("unpin crash")

        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)

        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page._pin_project("/path/p1")
            mock_crit.assert_called_with(page, "Pin Error", "Failed to pin project: pin crash")

        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page._unpin_project("/path/p1")
            mock_crit.assert_called_with(page, "Unpin Error", "Failed to unpin project: unpin crash")

    def test_unexpected_exceptions_handling_in_actions(self) -> None:
        """Verify unexpected non-orchestrator exceptions are caught and surfaced."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = []
        mock_orch.create_project.side_effect = RuntimeError("unexpected creation fail")
        mock_orch.open_project.side_effect = RuntimeError("unexpected open fail")

        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)

        # 1. Creation unexpected fail
        with patch("frontend.pages.project_dashboard.NewProjectDialog") as MockDialog:
            dialog_instance = MockDialog.return_value
            dialog_instance.exec.return_value = QDialog.DialogCode.Accepted
            dialog_instance.get_metadata.return_value = _make_mock_workspace().metadata
            dialog_instance.get_workspace_folder.return_value = "/fail/path"
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                page._on_new_project()
                mock_crit.assert_called_with(page, "Unexpected Error", "Failed to create project: unexpected creation fail")

        # 2. Open unexpected fail
        with patch("PySide6.QtWidgets.QFileDialog.getExistingDirectory", return_value="/fail/path"):
            with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
                page._on_open_project()
                mock_crit.assert_called_with(page, "Unexpected Error", "Failed to open project: unexpected open fail")

    def test_list_recent_projects_failure(self) -> None:
        """Verify registry errors during refresh display a QMessageBox critical dialog."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.side_effect = RuntimeError("registry file missing")

        with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_crit:
            page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
            mock_crit.assert_called_with(page, "Load Error", "Failed to list recent projects: registry file missing")

    def test_selection_changed_no_user_data(self) -> None:
        """Verify selection change does not update details if item contains no UserRole data."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = []
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        # Add a dummy item without user role data
        dummy_item = QListWidgetItem("Dummy Project", page.recent_list)
        page.recent_list.addItem(dummy_item)
        page.recent_list.setCurrentItem(dummy_item)
        
        assert page.details_stack.currentIndex() == 0

    def test_context_menu_fallbacks(self) -> None:
        """Verify context menu request returns early on invalid items or missing data."""
        app_obj = DeployAIApplication()
        mock_orch = MagicMock()
        mock_orch.list_recent_projects.return_value = []
        
        page = ProjectDashboardPage(orchestrator=mock_orch, parent=app_obj.main_window)
        
        # Case 1: itemAt returns None
        with patch.object(page.recent_list, "itemAt", return_value=None):
            with patch("frontend.pages.project_dashboard.QMenu") as MockMenu:
                page._show_context_menu(QPoint(0, 0))
                MockMenu.assert_not_called()

        # Case 2: item exists but has no UserRole data
        dummy_item = QListWidgetItem("Dummy")
        with patch.object(page.recent_list, "itemAt", return_value=dummy_item):
            with patch("frontend.pages.project_dashboard.QMenu") as MockMenu:
                page._show_context_menu(QPoint(0, 0))
                MockMenu.assert_not_called()

