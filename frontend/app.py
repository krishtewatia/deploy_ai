"""DeployAI Application Bootstrap."""

import sys
from PySide6.QtWidgets import QApplication
from frontend.main_window import DeployAIMainWindow


class DeployAIApplication:
    """The PySide6 application bootstrap for DeployAI."""

    _instance = None
    _allow_qapp_reuse = False

    def __init__(self, workspace_orchestrator=None) -> None:
        if DeployAIApplication._instance is not None:
            raise RuntimeError("DeployAIApplication instance already exists.")

        q_app = QApplication.instance()
        if q_app is not None:
            if not self._allow_qapp_reuse:
                raise RuntimeError("QApplication already exists.")
            self.app = q_app
        else:
            self.app = QApplication(sys.argv if sys.argv else [])
        
        # Instantiate/inject workspace orchestrator
        if workspace_orchestrator is not None:
            self.orchestrator = workspace_orchestrator
        else:
            from backend.app.workspace import WorkspaceManager, RecentProjectsRegistry, WorkspaceOrchestrator
            from pathlib import Path
            manager = WorkspaceManager()
            reg_dir = Path.home() / ".deploy_ai"
            reg_dir.mkdir(parents=True, exist_ok=True)
            registry = RecentProjectsRegistry(str(reg_dir))
            self.orchestrator = WorkspaceOrchestrator(manager, registry)

        # Initialize and own MainWindow
        self.main_window = DeployAIMainWindow(self.orchestrator)

        
        # Set the singleton class instance
        DeployAIApplication._instance = self

    def run(self) -> int:
        """Start the QApplication event loop and show the main window."""
        self.main_window.show()
        return self.app.exec()

    @classmethod
    def _reset(cls) -> None:
        """Internal helper to reset singleton state for testing purposes."""
        import gc
        cls._instance = None
        q_app = QApplication.instance()
        if q_app is not None:
            q_app.closeAllWindows()
            q_app.quit()
            q_app.processEvents()
            del q_app
        gc.collect()
