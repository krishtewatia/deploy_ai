"""DeployAI shell widget layout implementing the layout regions and QStackedWidget page container."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QStackedWidget, QMainWindow
from PySide6.QtCore import Qt
from frontend.navigation import NavigationController


class DeployAIShell(QWidget):
    """The permanent shell container widget combining navigation sidebar and QStackedWidget container."""

    def __init__(self, parent: QWidget | None = None, orchestrator=None) -> None:
        super().__init__(parent)


        # Main layout is horizontal: left sidebar and right QStackedWidget
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Navigation Sidebar
        self.sidebar = QWidget(self)
        self.sidebar.setFixedWidth(260)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(8)

        # Placeholders buttons in the specified order
        button_names = [
            "Dashboard",
            "Projects",
            "Datasets",
            "Training",
            "Reports",
            "Settings",
        ]
        self.nav_buttons = []
        for name in button_names:
            btn = QPushButton(name, self.sidebar)
            btn.setEnabled(True)  # Enable sidebar buttons
            sidebar_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        # Spacer to push buttons to the top
        sidebar_layout.addStretch()

        # 2. Central Content Area: QStackedWidget page container
        self.stacked_widget = QStackedWidget(self)

        # Assemble sidebar and stacked widget area
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.stacked_widget)

        # Find or fallback to QMainWindow parent
        main_win = parent
        while main_win is not None and not isinstance(main_win, QMainWindow):
            main_win = main_win.parent()

        if main_win is None:
            self.dummy_win = QMainWindow()
            main_win = self.dummy_win

        # Instantiate NavigationController
        self.nav_controller = NavigationController(main_win, self.stacked_widget, orchestrator)


        # Connect buttons to switch pages
        for btn, name in zip(self.nav_buttons, button_names):
            btn.clicked.connect(lambda checked=False, n=name: self.nav_controller.switch_to_page(n))

