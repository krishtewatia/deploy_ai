"""Comprehensive tests for the Workspace Orchestrator (Stage 11D).

Covers:
- create_project (workspace + registry)
- open_project (workspace + registry update)
- delete_project (registry removal + workspace deletion)
- list_recent_projects delegation
- pin_project / unpin_project delegation
- validate_project delegation
- Subsystem failure wrapping (WorkspaceOrchestratorError)
- Dependency injection validation
- Input validation
- Non-mutation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pytest

from backend.app.workspace.manager import WorkspaceManager, WorkspaceManagerError
from backend.app.workspace.orchestrator import (
    WorkspaceOrchestrator,
    WorkspaceOrchestratorError,
)
from backend.app.workspace.recent_registry import (
    RecentProjectEntry,
    RecentProjectsRegistry,
    RecentProjectsRegistryError,
)
from backend.app.workspace.schemas import (
    ProjectMetadata,
    ProjectStatus,
    ProjectType,
    ProjectWorkspace,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    reg = tmp_path / "registry"
    reg.mkdir()
    return reg


@pytest.fixture
def manager() -> WorkspaceManager:
    return WorkspaceManager()


@pytest.fixture
def registry(registry_dir: Path) -> RecentProjectsRegistry:
    return RecentProjectsRegistry(str(registry_dir))


@pytest.fixture
def orchestrator(
    manager: WorkspaceManager,
    registry: RecentProjectsRegistry,
) -> WorkspaceOrchestrator:
    return WorkspaceOrchestrator(manager, registry)


@pytest.fixture
def metadata() -> ProjectMetadata:
    return ProjectMetadata(
        project_name="Test Project",
        project_type=ProjectType.CLASSIFICATION,
        created_timestamp="2025-06-15T10:00:00Z",
        description="Test description.",
        tags=["ml", "test"],
    )


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "my_workspace"


# -----------------------------------------------------------------------
# Dependency injection tests
# -----------------------------------------------------------------------

class TestDependencyInjection:

    def test_rejects_invalid_manager(self, registry: RecentProjectsRegistry) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="WorkspaceManager"):
            WorkspaceOrchestrator("not_a_manager", registry)  # type: ignore[arg-type]

    def test_rejects_none_manager(self, registry: RecentProjectsRegistry) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="WorkspaceManager"):
            WorkspaceOrchestrator(None, registry)  # type: ignore[arg-type]

    def test_rejects_invalid_registry(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="RecentProjectsRegistry"):
            WorkspaceOrchestrator(manager, "not_a_registry")  # type: ignore[arg-type]

    def test_rejects_none_registry(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="RecentProjectsRegistry"):
            WorkspaceOrchestrator(manager, None)  # type: ignore[arg-type]

    def test_valid_construction(
        self, manager: WorkspaceManager, registry: RecentProjectsRegistry
    ) -> None:
        orch = WorkspaceOrchestrator(manager, registry)
        assert isinstance(orch, WorkspaceOrchestrator)


# -----------------------------------------------------------------------
# create_project tests
# -----------------------------------------------------------------------

class TestCreateProject:

    def test_creates_workspace_and_registers(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        ws = orchestrator.create_project(str(workspace_root), metadata)
        assert isinstance(ws, ProjectWorkspace)
        assert workspace_root.is_dir()

        # Verify registered in recent projects
        projects = orchestrator.list_recent_projects()
        assert len(projects) == 1
        assert projects[0].project_id == ws.workspace_id

    def test_returns_project_workspace(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        ws = orchestrator.create_project(str(workspace_root), metadata)
        assert ws.metadata.project_name == "Test Project"
        assert ws.status == "ACTIVE"

    def test_wraps_manager_error(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        with pytest.raises(WorkspaceOrchestratorError, match="Workspace creation failed"):
            orchestrator.create_project(str(workspace_root), metadata)

    def test_wraps_registry_error(
        self,
        manager: WorkspaceManager,
        metadata: ProjectMetadata,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reg_dir = tmp_path / "reg"
        reg_dir.mkdir()
        registry = RecentProjectsRegistry(str(reg_dir))
        orch = WorkspaceOrchestrator(manager, registry)

        def _failing_add(*args: object, **kwargs: object) -> None:
            raise RecentProjectsRegistryError("registry broken")

        monkeypatch.setattr(registry, "add_workspace", _failing_add)
        ws_root = tmp_path / "ws"
        with pytest.raises(WorkspaceOrchestratorError, match="Registry update failed"):
            orch.create_project(str(ws_root), metadata)


# -----------------------------------------------------------------------
# open_project tests
# -----------------------------------------------------------------------

class TestOpenProject:

    def test_opens_and_updates_registry(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        time.sleep(0.01)

        ws = orchestrator.open_project(str(workspace_root))
        assert isinstance(ws, ProjectWorkspace)

        # Should have updated timestamp (re-registered)
        projects = orchestrator.list_recent_projects()
        assert len(projects) == 1

    def test_wraps_manager_error(
        self, orchestrator: WorkspaceOrchestrator, tmp_path: Path
    ) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="Workspace open failed"):
            orchestrator.open_project(str(tmp_path / "nonexistent"))

    def test_wraps_registry_error(
        self,
        manager: WorkspaceManager,
        metadata: ProjectMetadata,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        reg_dir = tmp_path / "reg"
        reg_dir.mkdir()
        registry = RecentProjectsRegistry(str(reg_dir))
        orch = WorkspaceOrchestrator(manager, registry)

        ws_root = tmp_path / "ws"
        manager.create_workspace(str(ws_root), metadata)

        def _failing_add(*args: object, **kwargs: object) -> None:
            raise RecentProjectsRegistryError("registry broken")

        monkeypatch.setattr(registry, "add_workspace", _failing_add)
        with pytest.raises(WorkspaceOrchestratorError, match="Registry update failed"):
            orch.open_project(str(ws_root))


# -----------------------------------------------------------------------
# delete_project tests
# -----------------------------------------------------------------------

class TestDeleteProject:

    def test_deletes_workspace_and_registry_entry(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        assert len(orchestrator.list_recent_projects()) == 1

        orchestrator.delete_project(str(workspace_root))
        assert not workspace_root.exists()
        assert len(orchestrator.list_recent_projects()) == 0

    def test_tolerates_missing_registry_entry(
        self,
        orchestrator: WorkspaceOrchestrator,
        manager: WorkspaceManager,
        metadata: ProjectMetadata,
        tmp_path: Path,
    ) -> None:
        # Create workspace directly (bypassing registry)
        ws_root = tmp_path / "direct_ws"
        manager.create_workspace(str(ws_root), metadata)

        # Delete through orchestrator — no registry entry exists
        orchestrator.delete_project(str(ws_root))
        assert not ws_root.exists()

    def test_wraps_manager_error(
        self, orchestrator: WorkspaceOrchestrator, tmp_path: Path
    ) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="Workspace deletion failed"):
            orchestrator.delete_project(str(tmp_path / "ghost"))


# -----------------------------------------------------------------------
# list_recent_projects tests
# -----------------------------------------------------------------------

class TestListRecentProjects:

    def test_empty_initially(self, orchestrator: WorkspaceOrchestrator) -> None:
        assert orchestrator.list_recent_projects() == []

    def test_lists_created_projects(
        self,
        orchestrator: WorkspaceOrchestrator,
        metadata: ProjectMetadata,
        tmp_path: Path,
    ) -> None:
        for i in range(3):
            root = tmp_path / f"ws_{i}"
            orchestrator.create_project(str(root), metadata)
        assert len(orchestrator.list_recent_projects()) == 3

    def test_returns_recent_project_entries(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        entries = orchestrator.list_recent_projects()
        assert all(isinstance(e, RecentProjectEntry) for e in entries)


# -----------------------------------------------------------------------
# pin / unpin tests
# -----------------------------------------------------------------------

class TestPinUnpin:

    def test_pin_project(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        orchestrator.pin_project(str(workspace_root))
        assert orchestrator.list_recent_projects()[0].pinned is True

    def test_unpin_project(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        orchestrator.pin_project(str(workspace_root))
        orchestrator.unpin_project(str(workspace_root))
        assert orchestrator.list_recent_projects()[0].pinned is False

    def test_pin_wraps_error(self, orchestrator: WorkspaceOrchestrator) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="Pin operation failed"):
            orchestrator.pin_project("/nonexistent")

    def test_unpin_wraps_error(self, orchestrator: WorkspaceOrchestrator) -> None:
        with pytest.raises(WorkspaceOrchestratorError, match="Unpin operation failed"):
            orchestrator.unpin_project("/nonexistent")


# -----------------------------------------------------------------------
# validate_project tests
# -----------------------------------------------------------------------

class TestValidateProject:

    def test_valid_workspace(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        orchestrator.create_project(str(workspace_root), metadata)
        assert orchestrator.validate_project(str(workspace_root)) is True

    def test_invalid_missing_workspace(
        self, orchestrator: WorkspaceOrchestrator, tmp_path: Path
    ) -> None:
        assert orchestrator.validate_project(str(tmp_path / "nope")) is False

    def test_wraps_manager_error(
        self,
        manager: WorkspaceManager,
        registry: RecentProjectsRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        orch = WorkspaceOrchestrator(manager, registry)

        def _failing_validate(*args: object, **kwargs: object) -> bool:
            raise WorkspaceManagerError("validate boom")

        monkeypatch.setattr(manager, "validate_workspace", _failing_validate)
        with pytest.raises(WorkspaceOrchestratorError, match="Workspace validation failed"):
            orch.validate_project("/some/path")


# -----------------------------------------------------------------------
# Non-mutation tests
# -----------------------------------------------------------------------

class TestNonMutation:

    def test_metadata_not_mutated(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        original_name = metadata.project_name
        original_type = metadata.project_type
        orchestrator.create_project(str(workspace_root), metadata)
        assert metadata.project_name == original_name
        assert metadata.project_type == original_type

    def test_create_and_open_return_different_objects(
        self,
        orchestrator: WorkspaceOrchestrator,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        created = orchestrator.create_project(str(workspace_root), metadata)
        opened = orchestrator.open_project(str(workspace_root))
        assert created is not opened
        assert created.workspace_id == opened.workspace_id


# -----------------------------------------------------------------------
# Error type tests
# -----------------------------------------------------------------------

class TestErrorType:

    def test_orchestrator_error_is_exception(self) -> None:
        assert issubclass(WorkspaceOrchestratorError, Exception)

    def test_error_preserves_cause(
        self, orchestrator: WorkspaceOrchestrator, tmp_path: Path
    ) -> None:
        with pytest.raises(WorkspaceOrchestratorError) as exc_info:
            orchestrator.open_project(str(tmp_path / "missing"))
        assert exc_info.value.__cause__ is not None
