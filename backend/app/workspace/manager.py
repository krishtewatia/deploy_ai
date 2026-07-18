"""Workspace Manager — filesystem operations for DeployAI project workspaces.

Responsible ONLY for creating, opening, validating, and deleting workspaces.
NOT responsible for SQLite, UI, or ML execution.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.workspace.schemas import (
    DEFAULT_FOLDERS,
    ProjectMetadata,
    ProjectStatus,
    ProjectWorkspace,
    WorkspaceFolder,
)

_PROJECT_FILE = "project.json"


class WorkspaceManagerError(Exception):
    """Raised when a workspace filesystem operation fails."""


class WorkspaceManager:
    """Manages the lifecycle of DeployAI project workspaces on disk.

    Operations
    ----------
    create_workspace  — scaffold a new workspace directory tree
    open_workspace    — load an existing workspace from disk
    workspace_exists  — check whether a workspace root contains project.json
    delete_workspace  — recursively remove a workspace directory
    validate_workspace — verify project.json + all required folders exist
    """

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    def create_workspace(
        self,
        root_directory: str,
        metadata: ProjectMetadata,
    ) -> ProjectWorkspace:
        """Create a new workspace directory tree with project.json.

        Parameters
        ----------
        root_directory:
            Filesystem path where the workspace will be scaffolded.
        metadata:
            Validated project metadata to persist.

        Returns
        -------
        ProjectWorkspace
            A freshly-constructed, immutable workspace object.

        Raises
        ------
        WorkspaceManagerError
            On empty path, invalid metadata, existing workspace, or
            any filesystem failure.
        """
        self._reject_empty_path(root_directory)
        self._reject_invalid_metadata(metadata)

        root = Path(root_directory).resolve()

        if (root / _PROJECT_FILE).exists():
            raise WorkspaceManagerError(
                f"Workspace already exists at '{root}'"
            )

        # Build the workspace schema object first (before touching disk)
        workspace = self._build_workspace(root, metadata)

        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Directory exists but has no project.json — allowed
            if not root.is_dir():
                raise WorkspaceManagerError(
                    f"Path exists and is not a directory: '{root}'"
                )
        except OSError as exc:
            raise WorkspaceManagerError(
                f"Failed to create workspace directory '{root}': {exc}"
            ) from exc

        # Create sub-folders
        for folder in workspace.folders:
            folder_path = root / folder.relative_path
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise WorkspaceManagerError(
                    f"Failed to create folder '{folder_path}': {exc}"
                ) from exc

        # Serialize project.json
        self._write_project_file(root, workspace)

        return workspace

    # ------------------------------------------------------------------
    # open
    # ------------------------------------------------------------------

    def open_workspace(self, root_directory: str) -> ProjectWorkspace:
        """Open and return an existing workspace from disk.

        Parameters
        ----------
        root_directory:
            Filesystem path of the workspace root.

        Returns
        -------
        ProjectWorkspace

        Raises
        ------
        WorkspaceManagerError
            If the root or project.json doesn't exist, or if
            deserialization / validation fails.
        """
        self._reject_empty_path(root_directory)

        root = Path(root_directory).resolve()

        if not root.is_dir():
            raise WorkspaceManagerError(
                f"Workspace root does not exist: '{root}'"
            )

        project_file = root / _PROJECT_FILE
        if not project_file.is_file():
            raise WorkspaceManagerError(
                f"Missing {_PROJECT_FILE} in '{root}'"
            )

        return self._read_project_file(root)

    # ------------------------------------------------------------------
    # workspace_exists
    # ------------------------------------------------------------------

    def workspace_exists(self, root_directory: str) -> bool:
        """Return True if root_directory contains a valid project.json.

        Parameters
        ----------
        root_directory:
            Filesystem path to check.

        Returns
        -------
        bool
        """
        self._reject_empty_path(root_directory)

        root = Path(root_directory).resolve()
        return root.is_dir() and (root / _PROJECT_FILE).is_file()

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete_workspace(self, root_directory: str) -> None:
        """Recursively delete a workspace directory.

        Parameters
        ----------
        root_directory:
            Filesystem path of the workspace to remove.

        Raises
        ------
        WorkspaceManagerError
            If the workspace doesn't exist or deletion fails.
        """
        self._reject_empty_path(root_directory)

        root = Path(root_directory).resolve()

        if not root.is_dir():
            raise WorkspaceManagerError(
                f"Workspace does not exist: '{root}'"
            )

        try:
            shutil.rmtree(root)
        except OSError as exc:
            raise WorkspaceManagerError(
                f"Failed to delete workspace '{root}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def validate_workspace(self, root_directory: str) -> bool:
        """Validate that a workspace has project.json and all required folders.

        Parameters
        ----------
        root_directory:
            Filesystem path to validate.

        Returns
        -------
        bool
            True only if project.json exists, deserializes correctly,
            metadata validates, and every required folder is present.
        """
        self._reject_empty_path(root_directory)

        root = Path(root_directory).resolve()

        if not root.is_dir():
            return False

        project_file = root / _PROJECT_FILE
        if not project_file.is_file():
            return False

        # Try to deserialize and validate
        try:
            workspace = self._read_project_file(root)
        except WorkspaceManagerError:
            return False

        # Check all declared folders exist
        for folder in workspace.folders:
            folder_path = root / folder.relative_path
            if not folder_path.is_dir():
                return False

        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_empty_path(root_directory: str) -> None:
        """Raise if root_directory is empty or whitespace."""
        if not isinstance(root_directory, str):
            raise WorkspaceManagerError("root_directory must be a string")
        if not root_directory.strip():
            raise WorkspaceManagerError(
                "root_directory cannot be empty or whitespace-only"
            )

    @staticmethod
    def _reject_invalid_metadata(metadata: ProjectMetadata) -> None:
        """Raise if metadata is not a valid ProjectMetadata."""
        if not isinstance(metadata, ProjectMetadata):
            raise WorkspaceManagerError(
                "metadata must be a ProjectMetadata instance"
            )

    @staticmethod
    def _build_workspace(
        root: Path,
        metadata: ProjectMetadata,
    ) -> ProjectWorkspace:
        """Construct a new ProjectWorkspace object (no IO)."""
        workspace_id = str(uuid.uuid4())
        folders = [f.model_copy() for f in DEFAULT_FOLDERS]
        return ProjectWorkspace(
            workspace_id=workspace_id,
            status=ProjectStatus.ACTIVE,
            metadata=metadata,
            root_path=str(root),
            folders=folders,
        )

    @staticmethod
    def _write_project_file(root: Path, workspace: ProjectWorkspace) -> None:
        """Serialize workspace to project.json."""
        project_file = root / _PROJECT_FILE
        try:
            raw = workspace.model_dump_json(indent=2)
            project_file.write_text(raw, encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            raise WorkspaceManagerError(
                f"Failed to write {_PROJECT_FILE}: {exc}"
            ) from exc

    @staticmethod
    def _read_project_file(root: Path) -> ProjectWorkspace:
        """Deserialize project.json to a ProjectWorkspace."""
        project_file = root / _PROJECT_FILE
        try:
            raw = project_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            return ProjectWorkspace.model_validate(data)
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkspaceManagerError(
                f"Failed to read {_PROJECT_FILE}: {exc}"
            ) from exc
        except Exception as exc:
            raise WorkspaceManagerError(
                f"Invalid workspace data in {_PROJECT_FILE}: {exc}"
            ) from exc
