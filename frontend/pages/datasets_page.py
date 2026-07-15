"""Datasets page for DeployAI."""




from frontend.pages.dataset_manager import DatasetManagerPage


class DatasetsPage(DatasetManagerPage):
    """The Datasets page inheriting from DatasetManagerPage."""

    def __init__(self, orchestrator=None, parent=None) -> None:
        super().__init__(orchestrator=orchestrator, parent=parent)

