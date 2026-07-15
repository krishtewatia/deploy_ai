"""DeployAI Main Window implementation."""

from PySide6.QtWidgets import QMainWindow, QToolBar
from PySide6.QtGui import QGuiApplication, QIcon
from frontend.shell import DeployAIShell


class DeployAIMainWindow(QMainWindow):
    """The main application window for DeployAI."""

    def __init__(self, orchestrator=None) -> None:
        super().__init__()
        self.orchestrator = orchestrator
        self.setWindowTitle("DeployAI")
        self.setMinimumSize(1200, 800)
        self._center_on_screen()
        self._init_menus()
        self._init_toolbar()
        self._init_shell()
        self._init_status_bar()
        self._init_icon()


    def _center_on_screen(self) -> None:
        """Center the window on the primary screen."""
        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geo = screen.geometry()
            width = self.width() if self.width() > 0 else 1200
            height = self.height() if self.height() > 0 else 800
            x = max(0, (screen_geo.width() - width) // 2)
            y = max(0, (screen_geo.height() - height) // 2)
            self.move(x, y)

    def _init_menus(self) -> None:
        """Initialize menu bar with File and Help placeholders."""
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu("File")
        self.help_menu = menu_bar.addMenu("Help")

    def _init_toolbar(self) -> None:
        """Initialize empty main window toolbar docked at the top."""
        self.toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(self.toolbar)

    def _init_shell(self) -> None:
        """Set the central widget to DeployAIShell."""
        self.shell = DeployAIShell(self, self.orchestrator)
        self.setCentralWidget(self.shell)


    def _init_status_bar(self) -> None:
        """Initialize status bar with 'Ready' message."""
        status_bar = self.statusBar()
        status_bar.showMessage("Ready")

    def _init_icon(self) -> None:
        """Set placeholder application icon."""
        self.setWindowIcon(QIcon("assets/placeholder_icon.png"))

