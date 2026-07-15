"""DeployAI Navigation Controller implementation."""

from PySide6.QtWidgets import QStackedWidget, QMainWindow
from frontend.pages import (
    DashboardPage,
    ProjectsPage,
    DatasetsPage,
    TrainingPage,
    ReportsPage,
    SettingsPage,
)


class NavigationController:
    """Manages page routing and window title updates for DeployAI."""

    def __init__(self, main_window: QMainWindow, stacked_widget: QStackedWidget, orchestrator=None) -> None:
        self.main_window = main_window
        self.stacked_widget = stacked_widget

        # Exactly six pages are instantiated once
        self.pages = {
            "Dashboard": DashboardPage(),
            "Projects": ProjectsPage(orchestrator=orchestrator),
            "Datasets": DatasetsPage(orchestrator=orchestrator),
            "Training": TrainingPage(orchestrator=orchestrator),
            "Reports": ReportsPage(),
            "Settings": SettingsPage(),
        }


        # Clear any existing items in QStackedWidget to guarantee clean state
        while self.stacked_widget.count() > 0:
            widget = self.stacked_widget.widget(0)
            self.stacked_widget.removeWidget(widget)

        # Register pages to QStackedWidget
        for page in self.pages.values():
            self.stacked_widget.addWidget(page)

        # Set default page to Dashboard
        self.switch_to_page("Dashboard")

    def switch_to_page(self, name: str) -> None:
        """Switch active page and update the main window title."""
        if name not in self.pages:
            raise ValueError(f"Unknown page registration: {name}")
        page = self.pages[name]
        self.stacked_widget.setCurrentWidget(page)
        self.main_window.setWindowTitle(f"DeployAI — {name}")
