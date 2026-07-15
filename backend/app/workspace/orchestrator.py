"""Workspace Orchestrator — single high-level entry point for all workspace operations.

The UI interacts ONLY with this class.  It delegates to
``WorkspaceManager`` and ``RecentProjectsRegistry``.

It does NOT implement SQLite, UI, settings, or launcher logic.
"""

from __future__ import annotations

from typing import List

from backend.app.workspace.manager import WorkspaceManager, WorkspaceManagerError
from backend.app.workspace.recent_registry import (
    RecentProjectEntry,
    RecentProjectsRegistry,
    RecentProjectsRegistryError,
)
from backend.app.workspace.schemas import ProjectMetadata, ProjectWorkspace


class WorkspaceOrchestratorError(Exception):
    """Raised when an orchestrated workspace operation fails."""


class WorkspaceOrchestrator:
    """Coordinates workspace lifecycle through the manager and registry.

    Parameters
    ----------
    workspace_manager : WorkspaceManager
        Injected manager for filesystem workspace operations.
    recent_registry : RecentProjectsRegistry
        Injected registry for tracking recently opened projects.
    """

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        recent_registry: RecentProjectsRegistry,
    ) -> None:
        if not isinstance(workspace_manager, WorkspaceManager):
            raise WorkspaceOrchestratorError(
                "workspace_manager must be a WorkspaceManager instance"
            )
        if not isinstance(recent_registry, RecentProjectsRegistry):
            raise WorkspaceOrchestratorError(
                "recent_registry must be a RecentProjectsRegistry instance"
            )
        self._manager = workspace_manager
        self._registry = recent_registry

    # ------------------------------------------------------------------
    # create_project
    # ------------------------------------------------------------------

    def create_project(
        self,
        root_directory: str,
        metadata: ProjectMetadata,
    ) -> ProjectWorkspace:
        """Create a new workspace and register it in the recent-projects list.

        1. Create workspace via WorkspaceManager
        2. Register project in RecentProjectsRegistry
        3. Return the ProjectWorkspace

        Raises
        ------
        WorkspaceOrchestratorError
            If workspace creation or registry update fails.
        """
        try:
            workspace = self._manager.create_workspace(root_directory, metadata)
        except WorkspaceManagerError as exc:
            raise WorkspaceOrchestratorError(
                f"Workspace creation failed: {exc}"
            ) from exc

        try:
            self._registry.add_workspace(workspace)
        except RecentProjectsRegistryError as exc:
            raise WorkspaceOrchestratorError(
                f"Registry update failed: {exc}"
            ) from exc

        return workspace

    # ------------------------------------------------------------------
    # open_project
    # ------------------------------------------------------------------

    def open_project(self, root_directory: str) -> ProjectWorkspace:
        """Open an existing workspace and update the recent-projects list.

        1. Open workspace via WorkspaceManager
        2. Update recent registry timestamp
        3. Return the ProjectWorkspace

        Raises
        ------
        WorkspaceOrchestratorError
            If opening the workspace or registry update fails.
        """
        try:
            workspace = self._manager.open_workspace(root_directory)
        except WorkspaceManagerError as exc:
            raise WorkspaceOrchestratorError(
                f"Workspace open failed: {exc}"
            ) from exc

        try:
            self._registry.add_workspace(workspace)
        except RecentProjectsRegistryError as exc:
            raise WorkspaceOrchestratorError(
                f"Registry update failed: {exc}"
            ) from exc

        return workspace

    # ------------------------------------------------------------------
    # delete_project
    # ------------------------------------------------------------------

    def delete_project(self, root_directory: str) -> None:
        """Remove a project from the registry and delete the workspace.

        1. Remove registry entry (tolerates missing entry)
        2. Delete workspace from disk

        Raises
        ------
        WorkspaceOrchestratorError
            If workspace deletion fails.
        """
        try:
            self._registry.remove_workspace(root_directory)
        except RecentProjectsRegistryError:
            # Entry may not exist in registry — tolerate this
            pass

        try:
            self._manager.delete_workspace(root_directory)
        except WorkspaceManagerError as exc:
            raise WorkspaceOrchestratorError(
                f"Workspace deletion failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # list_recent_projects
    # ------------------------------------------------------------------

    def list_recent_projects(self) -> List[RecentProjectEntry]:
        """Return the sorted recent-projects list (pinned first)."""
        return self._registry.list_projects()

    # ------------------------------------------------------------------
    # pin_project / unpin_project
    # ------------------------------------------------------------------

    def pin_project(self, workspace_path: str) -> None:
        """Pin a project so it stays at the top of the recent list.

        Raises
        ------
        WorkspaceOrchestratorError
            If the registry operation fails.
        """
        try:
            self._registry.pin_workspace(workspace_path)
        except RecentProjectsRegistryError as exc:
            raise WorkspaceOrchestratorError(
                f"Pin operation failed: {exc}"
            ) from exc

    def unpin_project(self, workspace_path: str) -> None:
        """Remove the pin from a project.

        Raises
        ------
        WorkspaceOrchestratorError
            If the registry operation fails.
        """
        try:
            self._registry.unpin_workspace(workspace_path)
        except RecentProjectsRegistryError as exc:
            raise WorkspaceOrchestratorError(
                f"Unpin operation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # validate_project
    # ------------------------------------------------------------------

    def validate_project(self, workspace_path: str) -> bool:
        """Validate that a workspace directory is structurally complete.

        Returns
        -------
        bool
            True if project.json exists, metadata validates, and all
            required folders are present.

        Raises
        ------
        WorkspaceOrchestratorError
            If validation encounters an unexpected error.
        """
        try:
            return self._manager.validate_workspace(workspace_path)
        except WorkspaceManagerError as exc:
            raise WorkspaceOrchestratorError(
                f"Workspace validation failed: {exc}"
            ) from exc
