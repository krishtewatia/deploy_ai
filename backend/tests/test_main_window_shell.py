"""Tests for the main window shell and layout structure (Stage 12B)."""

import os
import pytest
from PySide6.QtWidgets import QMainWindow, QToolBar, QWidget, QPushButton, QLabel, QStackedWidget
from PySide6.QtCore import QSize
from frontend.app import DeployAIApplication
from frontend.main_window import DeployAIMainWindow
from frontend.shell import DeployAIShell


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup and clean up the QApplication reuse state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def test_main_window_owns_shell() -> None:
    """Test that the QMainWindow contains the DeployAIShell as the central widget."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    assert hasattr(win, "shell")
    assert isinstance(win.shell, DeployAIShell)
    assert win.centralWidget() is win.shell


def test_toolbar_exists() -> None:
    """Test that exactly one toolbar is created and docked at the top."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    assert hasattr(win, "toolbar")
    assert isinstance(win.toolbar, QToolBar)
    
    # Check that it exists inside the main window
    toolbars = win.findChildren(QToolBar)
    assert len(toolbars) == 1
    assert toolbars[0] is win.toolbar


def test_sidebar_exists_and_dimensions() -> None:
    """Test that the navigation sidebar exists and has a fixed width of 260px."""
    app_obj = DeployAIApplication()
    shell = app_obj.main_window.shell
    assert hasattr(shell, "sidebar")
    assert isinstance(shell.sidebar, QWidget)
    assert shell.sidebar.minimumWidth() == 260
    assert shell.sidebar.maximumWidth() == 260


def test_button_count_order_and_enabled() -> None:
    """Test that exactly 6 enabled buttons exist in the declared order."""
    app_obj = DeployAIApplication()
    shell = app_obj.main_window.shell
    expected_order = [
        "Dashboard",
        "Projects",
        "Datasets",
        "Training",
        "Reports",
        "Settings",
    ]
    assert len(shell.nav_buttons) == len(expected_order)
    for idx, name in enumerate(expected_order):
        btn = shell.nav_buttons[idx]
        assert isinstance(btn, QPushButton)
        assert btn.text() == name
        assert btn.isEnabled()  # Enabled in 12C


def test_central_placeholder_replaced_by_stacked_widget() -> None:
    """Test that the central placeholder is replaced by QStackedWidget page container."""
    app_obj = DeployAIApplication()
    shell = app_obj.main_window.shell
    
    assert hasattr(shell, "stacked_widget")
    assert isinstance(shell.stacked_widget, QStackedWidget)



def test_status_bar_preserved() -> None:
    """Test that the status bar is preserved and initialized correctly."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    status_bar = win.statusBar()
    assert status_bar is not None
    assert status_bar.currentMessage() == "Ready"


def test_no_forbidden_imports() -> None:
    """Verify that no frontend source files import backend, ML, or workspace modules except backend.app.workspace."""
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
                        assert "import workspace" not in line
                        assert "from workspace" not in line


