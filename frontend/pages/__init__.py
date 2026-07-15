"""DeployAI Pages package."""

from frontend.pages.dashboard_page import DashboardPage
from frontend.pages.projects_page import ProjectsPage
from frontend.pages.datasets_page import DatasetsPage
from frontend.pages.training_page import TrainingPage
from frontend.pages.reports_page import ReportsPage
from frontend.pages.settings_page import SettingsPage
from frontend.pages.project_dashboard import ProjectDashboardPage
from frontend.pages.dialogs.new_project_dialog import NewProjectDialog
from frontend.pages.dataset_manager import DatasetManagerPage
from frontend.pages.dialogs.import_dataset_dialog import ImportDatasetDialog
from frontend.pages.dataset_analysis import DatasetAnalysisWidget
from frontend.pages.planning_wizard import MLPlanningWizard
from frontend.pages.training_execution import TrainingExecutionPage
from frontend.pages.workers.training_worker import TrainingWorker
from frontend.pages.training_results import TrainingResultsPage

__all__ = [
    "DashboardPage",
    "ProjectsPage",
    "DatasetsPage",
    "TrainingPage",
    "ReportsPage",
    "SettingsPage",
    "ProjectDashboardPage",
    "NewProjectDialog",
    "DatasetManagerPage",
    "ImportDatasetDialog",
    "DatasetAnalysisWidget",
    "MLPlanningWizard",
    "TrainingExecutionPage",
    "TrainingWorker",
    "TrainingResultsPage",
]


from PySide6.QtWidgets import QVBoxLayout

# Monkey patch ReportsPage to containerize TrainingResultsPage layout automatically
_orig_reports_init = ReportsPage.__init__

def _new_reports_init(self, *args, **kwargs):
    _orig_reports_init(self, *args, **kwargs)
    if self.layout():
        while self.layout().count() > 0:
            item = self.layout().takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    else:
        self.setLayout(QVBoxLayout(self))
    self.layout().setContentsMargins(0, 0, 0, 0)
    self._results_widget = None

ReportsPage.__init__ = _new_reports_init

_orig_reports_show = ReportsPage.showEvent
def _new_reports_show(self, event):
    _orig_reports_show(self, event)
    main_win = self.window()
    orch = getattr(main_win, "orchestrator", None)
    if not self._results_widget:
        self._results_widget = TrainingResultsPage(orchestrator=orch, parent=self)
        self.layout().addWidget(self._results_widget)
    self._results_widget.refresh_results()

ReportsPage.showEvent = _new_reports_show


