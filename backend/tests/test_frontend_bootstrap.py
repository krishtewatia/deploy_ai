"""Tests for the frontend application bootstrap and main window (Stage 12A)."""

import sys
import subprocess
import pytest
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QSize
from frontend.app import DeployAIApplication
from frontend.main_window import DeployAIMainWindow


@pytest.fixture(autouse=True)
def setup_and_teardown() -> None:
    """Setup class attribute for test reuse and reset singleton state."""
    DeployAIApplication._allow_qapp_reuse = True
    DeployAIApplication._reset()
    yield
    DeployAIApplication._reset()
    DeployAIApplication._allow_qapp_reuse = False


def test_application_creation() -> None:
    """Test that DeployAIApplication can be created and instantiates the main window."""
    app_obj = DeployAIApplication()
    assert app_obj.app is not None
    assert isinstance(app_obj.main_window, DeployAIMainWindow)
    assert DeployAIApplication._instance is app_obj
    
    # Test injecting an orchestrator
    DeployAIApplication._reset()
    from unittest.mock import MagicMock
    mock_orch = MagicMock()
    mock_orch.list_recent_projects.return_value = []
    app_obj2 = DeployAIApplication(workspace_orchestrator=mock_orch)
    assert app_obj2.orchestrator is mock_orch




def test_window_type_and_title() -> None:
    """Test the window inherits QMainWindow and has correct title."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    assert isinstance(win, QMainWindow)
    assert win.windowTitle() == "DeployAI — Dashboard"



def test_window_minimum_size() -> None:
    """Test that the main window has the required minimum size of 1200 x 800."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    assert win.minimumSize() == QSize(1200, 800)


def test_window_centering() -> None:
    """Test that the window centering calculation executes without error."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    win._center_on_screen()
    assert win.x() >= 0
    assert win.y() >= 0


def test_menu_existence() -> None:
    """Test that 'File' and 'Help' menus exist as placeholders on the menu bar."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    assert win.file_menu is not None
    assert win.file_menu.title() == "File"
    assert win.help_menu is not None
    assert win.help_menu.title() == "Help"


def test_status_bar_existence() -> None:
    """Test that status bar is enabled and displays 'Ready' initially."""
    app_obj = DeployAIApplication()
    win = app_obj.main_window
    status_bar = win.statusBar()
    assert status_bar is not None
    assert status_bar.currentMessage() == "Ready"


def test_singleton_application_behavior_in_process() -> None:
    """Test that multiple DeployAIApplication creations in the same process raise RuntimeError."""
    # Ensure reuse is allowed for the first instance
    DeployAIApplication._allow_qapp_reuse = True
    app_obj = DeployAIApplication()
    
    # Try creating second instance, which should trigger the first check
    with pytest.raises(RuntimeError, match="DeployAIApplication instance already exists"):
        DeployAIApplication()


def test_singleton_qapp_behavior_in_process() -> None:
    """Test that DeployAIApplication rejects creation if QApplication exists and reuse is disabled."""
    # Ensure a QApplication exists (it already does, but let's make sure)
    assert QApplication.instance() is not None
    
    # Disable reuse to trigger the QApplication exists exception
    DeployAIApplication._allow_qapp_reuse = False
    with pytest.raises(RuntimeError, match="QApplication already exists"):
        DeployAIApplication()


def test_app_run_method() -> None:
    """Test the run method executing and exiting the event loop using a QTimer."""
    from PySide6.QtCore import QTimer
    app_obj = DeployAIApplication()
    
    # Schedule event loop termination immediately after it starts
    QTimer.singleShot(0, app_obj.app.quit)
    
    exit_code = app_obj.run()
    assert isinstance(exit_code, int)


def test_singleton_application_behavior_subprocess() -> None:
    """Test that multiple DeployAIApplication creations in the same process are rejected."""
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "c:\\Users\\hp\\Downloads\\deploy_ai"
    
    code = (
        "import sys\n"
        "from frontend.app import DeployAIApplication\n"
        "app1 = DeployAIApplication()\n"
        "try:\n"
        "    app2 = DeployAIApplication()\n"
        "    print('FAIL')\n"
        "except RuntimeError as e:\n"
        "    if 'DeployAIApplication instance already exists' in str(e):\n"
        "        print('SUCCESS')\n"
        "    else:\n"
        "        print('WRONG_ERROR')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0, f"Subprocess failed with stderr: {result.stderr}"
    assert result.stdout.strip() == "SUCCESS"


def test_singleton_qapp_behavior_subprocess() -> None:
    """Test that creation is rejected if a raw QApplication already exists."""
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "c:\\Users\\hp\\Downloads\\deploy_ai"

    code = (
        "import sys\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from frontend.app import DeployAIApplication\n"
        "qapp = QApplication([])\n"
        "try:\n"
        "    app = DeployAIApplication()\n"
        "    print('FAIL')\n"
        "except RuntimeError as e:\n"
        "    if 'QApplication already exists' in str(e):\n"
        "        print('SUCCESS')\n"
        "    else:\n"
        "        print('WRONG_ERROR')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 0, f"Subprocess failed with stderr: {result.stderr}"
    assert result.stdout.strip() == "SUCCESS"


