"""Tests for the application navigation controller and page routing (Stage 12C)."""

import os
import pytest
from PySide6.QtWidgets import QStackedWidget, QPushButton, QMainWindow
from frontend.app import DeployAIApplication
from frontend.main_window import DeployAIMainWindow
from frontend.shell import DeployAIShell
from frontend.navigation import NavigationController
from frontend.pages import (
    DashboardPage,
    ProjectsPage,
    DatasetsPage,
    TrainingPage,
    ReportsPage,
    SettingsPage,
)


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def test_pages_created_and_stacked_count() -> None:
    """Test that all six pages are registered exactly once inside the QStackedWidget."""
    app_obj = DeployAIApplication()
    shell = app_obj.main_window.shell
    stacked = shell.stacked_widget
    
    assert isinstance(stacked, QStackedWidget)
    assert stacked.count() == 6
    
    controller = shell.nav_controller
    assert isinstance(controller.pages["Dashboard"], DashboardPage)
    assert isinstance(controller.pages["Projects"], ProjectsPage)
    assert isinstance(controller.pages["Datasets"], DatasetsPage)
    assert isinstance(controller.pages["Training"], TrainingPage)
    assert isinstance(controller.pages["Reports"], ReportsPage)
    assert isinstance(controller.pages["Settings"], SettingsPage)


def test_buttons_count_order_and_enabled() -> None:
    """Test that the navigation buttons are correctly ordered and enabled."""
    app_obj = DeployAIApplication()
    shell = app_obj.main_window.shell
    
    expected_order = ["Dashboard", "Projects", "Datasets", "Training", "Reports", "Settings"]
    assert len(shell.nav_buttons) == 6
    for idx, name in enumerate(expected_order):
        btn = shell.nav_buttons[idx]
        assert btn.text() == name
        assert btn.isEnabled()


def test_default_page_and_title() -> None:
    """Test that Dashboard is the default active page and matches the window title."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    shell = win.shell
    
    assert shell.stacked_widget.currentWidget() is shell.nav_controller.pages["Dashboard"]
    assert win.windowTitle() == "DeployAI — Dashboard"


def test_switch_pages_updates_active_widget_and_title() -> None:
    """Test that switching pages correctly switches active widgets and titles."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    shell = win.shell
    controller = shell.nav_controller
    
    pages_to_test = ["Projects", "Datasets", "Training", "Reports", "Settings", "Dashboard"]
    for page_name in pages_to_test:
        controller.switch_to_page(page_name)
        assert shell.stacked_widget.currentWidget() is controller.pages[page_name]
        assert win.windowTitle() == f"DeployAI — {page_name}"


def test_button_clicks_trigger_navigation() -> None:
    """Test that clicking navigation buttons switches the active page."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    shell = win.shell
    
    # Click projects button (index 1)
    shell.nav_buttons[1].click()
    assert shell.stacked_widget.currentWidget() is shell.nav_controller.pages["Projects"]
    assert win.windowTitle() == "DeployAI — Projects"
    
    # Click settings button (index 5)
    shell.nav_buttons[5].click()
    assert shell.stacked_widget.currentWidget() is shell.nav_controller.pages["Settings"]
    assert win.windowTitle() == "DeployAI — Settings"


def test_navigation_controller_invalid_page() -> None:
    """Test that switching to an unregistered page raises ValueError."""
    app_obj = DeployAIApplication()
    controller = app_obj.main_window.shell.nav_controller
    with pytest.raises(ValueError, match="Unknown page registration"):
        controller.switch_to_page("InvalidPage")


def test_no_duplicate_registrations() -> None:
    """Test that NavigationController doesn't instantiate pages twice or register duplicates."""
    app_obj = DeployAIApplication()
    shell = app_obj.main_window.shell
    controller = shell.nav_controller
    
    unique_pages = set(controller.pages.values())
    assert len(unique_pages) == 6


def test_no_forbidden_imports() -> None:
    """Ensure that no frontend source files import backend or ML modules except backend.app.workspace."""
    frontend_dir = "c:\\Users\\hp\\Downloads\\deploy_ai\\frontend"
    for root, _, files in os.walk(frontend_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    for line in content.splitlines():
                        if "import" in line and "backend" in line:
                            assert "backend.app.workspace" in line
                        assert "import ml" not in line
                        assert "from ml" not in line




def test_navigation_controller_clearing_existing_widgets() -> None:
    """Verify QStackedWidget is cleared of existing widgets on controller initialization."""
    from PySide6.QtWidgets import QWidget, QStackedWidget
    win = QMainWindow()
    stacked = QStackedWidget()
    
    # Add some pre-existing widgets
    w1 = QWidget()
    w2 = QWidget()
    stacked.addWidget(w1)
    stacked.addWidget(w2)
    assert stacked.count() == 2
    
    # Init controller
    controller = NavigationController(win, stacked)
    # The pre-existing widgets should be cleared, leaving only the 6 registered pages
    assert stacked.count() == 6
    for i in range(stacked.count()):
        assert stacked.widget(i) is not w1
        assert stacked.widget(i) is not w2


def test_shell_parent_lookup_traversal_and_fallback() -> None:
    """Test parent lookup traversing widget hierarchy and falling back to dummy window."""
    from PySide6.QtWidgets import QWidget
    
    # Case 1: Parent is None (fallback to dummy)
    shell_none = DeployAIShell(None)
    assert hasattr(shell_none, "dummy_win")
    assert isinstance(shell_none.dummy_win, QMainWindow)
    
    # Case 2: Parent is a child of QMainWindow (traversal)
    win = QMainWindow()
    child_widget = QWidget(win)
    grandchild_widget = QWidget(child_widget)
    
    shell_child = DeployAIShell(grandchild_widget)
    assert shell_child.nav_controller.main_window is win

