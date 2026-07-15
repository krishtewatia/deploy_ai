"""Project Workspace package."""

from backend.app.workspace.schemas import (
    ProjectStatus,
    ProjectType,
    WorkspaceFolder,
    ProjectMetadata,
    ProjectWorkspace,
)
from backend.app.workspace.manager import (
    WorkspaceManager,
    WorkspaceManagerError,
)
from backend.app.workspace.recent_registry import (
    RecentProjectsRegistry,
    RecentProjectsRegistryError,
    RecentProjectEntry,
)
from backend.app.workspace.orchestrator import (
    WorkspaceOrchestrator,
    WorkspaceOrchestratorError,
)

__all__ = [
    "ProjectStatus",
    "ProjectType",
    "WorkspaceFolder",
    "ProjectMetadata",
    "ProjectWorkspace",
    "WorkspaceManager",
    "WorkspaceManagerError",
    "RecentProjectsRegistry",
    "RecentProjectsRegistryError",
    "RecentProjectEntry",
    "WorkspaceOrchestrator",
    "WorkspaceOrchestratorError",
]
