"""Reports page for DeployAI."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class ReportsPage(QWidget):
    """The Reports page display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel("Reports", self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.subtitle_label = QLabel("Coming Soon", self)
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
