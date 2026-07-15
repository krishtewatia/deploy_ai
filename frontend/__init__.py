"""DeployAI Frontend package."""

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
    ProjectDashboardPage,
    NewProjectDialog,
)

__all__ = [
    "DeployAIApplication",
    "DeployAIMainWindow",
    "DeployAIShell",
    "NavigationController",
    "DashboardPage",
    "ProjectsPage",
    "DatasetsPage",
    "TrainingPage",
    "ReportsPage",
    "SettingsPage",
    "ProjectDashboardPage",
    "NewProjectDialog",
]



