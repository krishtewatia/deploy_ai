"""Comprehensive tests for the Workspace Manager (Stage 11B).

Covers:
- Workspace creation (happy path, defaults, folders, project.json)
- Workspace opening (happy path, missing root, missing project.json)
- Workspace validation (valid, missing folders, corrupted JSON, invalid metadata)
- Workspace deletion (happy path, non-existent)
- workspace_exists (true/false)
- Duplicate creation rejection
- Non-mutation of inputs
- Round-trip serialization
- Error wrapping
- Edge cases (empty path, non-string path, whitespace path)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.workspace.manager import WorkspaceManager, WorkspaceManagerError
from backend.app.workspace.schemas import (
    DEFAULT_FOLDERS,
    ProjectMetadata,
    ProjectStatus,
    ProjectType,
    ProjectWorkspace,
)

_PROJECT_FILE = "project.json"


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def manager() -> WorkspaceManager:
    return WorkspaceManager()


@pytest.fixture
def metadata() -> ProjectMetadata:
    return ProjectMetadata(
        project_name="Test Project",
        project_type=ProjectType.CLASSIFICATION,
        created_timestamp="2025-06-15T10:00:00Z",
        description="A test project.",
        tags=["ml", "test"],
    )


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    """Return a path inside tmp_path that does NOT yet exist."""
    return tmp_path / "my_workspace"


@pytest.fixture
def created_workspace(
    manager: WorkspaceManager,
    workspace_root: Path,
    metadata: ProjectMetadata,
) -> ProjectWorkspace:
    """Create and return a workspace — convenience fixture."""
    return manager.create_workspace(str(workspace_root), metadata)


# -----------------------------------------------------------------------
# CREATE tests
# -----------------------------------------------------------------------

class TestCreateWorkspace:
    """Tests for WorkspaceManager.create_workspace."""

    def test_returns_project_workspace(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        ws = manager.create_workspace(str(workspace_root), metadata)
        assert isinstance(ws, ProjectWorkspace)

    def test_sets_active_status(self, created_workspace: ProjectWorkspace) -> None:
        assert created_workspace.status == "ACTIVE"

    def test_generates_uuid_workspace_id(self, created_workspace: ProjectWorkspace) -> None:
        assert len(created_workspace.workspace_id) == 36  # UUID format
        assert "-" in created_workspace.workspace_id

    def test_stores_metadata(
        self, created_workspace: ProjectWorkspace, metadata: ProjectMetadata
    ) -> None:
        assert created_workspace.metadata.project_name == metadata.project_name
        assert created_workspace.metadata.project_type == metadata.project_type
        assert created_workspace.metadata.created_timestamp == metadata.created_timestamp
        assert created_workspace.metadata.description == metadata.description
        assert created_workspace.metadata.tags == metadata.tags

    def test_creates_root_directory(self, workspace_root: Path, created_workspace: ProjectWorkspace) -> None:
        assert workspace_root.is_dir()

    def test_creates_project_json(self, workspace_root: Path, created_workspace: ProjectWorkspace) -> None:
        assert (workspace_root / _PROJECT_FILE).is_file()

    def test_creates_default_folders(self, workspace_root: Path, created_workspace: ProjectWorkspace) -> None:
        for folder in DEFAULT_FOLDERS:
            assert (workspace_root / folder.relative_path).is_dir()

    def test_project_json_is_valid_json(self, workspace_root: Path, created_workspace: ProjectWorkspace) -> None:
        raw = (workspace_root / _PROJECT_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["workspace_id"] == created_workspace.workspace_id

    def test_default_folder_count(self, created_workspace: ProjectWorkspace) -> None:
        assert len(created_workspace.folders) == 5

    def test_root_path_stored(
        self, workspace_root: Path, created_workspace: ProjectWorkspace
    ) -> None:
        assert created_workspace.root_path == str(workspace_root.resolve())

    def test_rejects_duplicate_creation(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        with pytest.raises(WorkspaceManagerError, match="already exists"):
            manager.create_workspace(str(workspace_root), metadata)

    def test_rejects_empty_path(
        self, manager: WorkspaceManager, metadata: ProjectMetadata
    ) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.create_workspace("", metadata)

    def test_rejects_whitespace_path(
        self, manager: WorkspaceManager, metadata: ProjectMetadata
    ) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.create_workspace("   ", metadata)

    def test_rejects_non_string_path(
        self, manager: WorkspaceManager, metadata: ProjectMetadata
    ) -> None:
        with pytest.raises(WorkspaceManagerError, match="string"):
            manager.create_workspace(123, metadata)  # type: ignore[arg-type]

    def test_rejects_invalid_metadata_none(
        self, manager: WorkspaceManager, workspace_root: Path
    ) -> None:
        with pytest.raises(WorkspaceManagerError, match="ProjectMetadata"):
            manager.create_workspace(str(workspace_root), None)  # type: ignore[arg-type]

    def test_rejects_invalid_metadata_wrong_type(
        self, manager: WorkspaceManager, workspace_root: Path
    ) -> None:
        with pytest.raises(WorkspaceManagerError, match="ProjectMetadata"):
            manager.create_workspace(str(workspace_root), {"not": "metadata"})  # type: ignore[arg-type]


# -----------------------------------------------------------------------
# OPEN tests
# -----------------------------------------------------------------------

class TestOpenWorkspace:
    """Tests for WorkspaceManager.open_workspace."""

    def test_opens_existing_workspace(
        self,
        manager: WorkspaceManager,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        original = manager.create_workspace(str(workspace_root), metadata)
        opened = manager.open_workspace(str(workspace_root))
        assert isinstance(opened, ProjectWorkspace)
        assert opened.workspace_id == original.workspace_id

    def test_preserves_metadata(
        self,
        manager: WorkspaceManager,
        workspace_root: Path,
        metadata: ProjectMetadata,
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        opened = manager.open_workspace(str(workspace_root))
        assert opened.metadata.project_name == metadata.project_name
        assert opened.metadata.project_type == metadata.project_type

    def test_preserves_status(
        self, created_workspace: ProjectWorkspace,
        manager: WorkspaceManager, workspace_root: Path,
    ) -> None:
        opened = manager.open_workspace(str(workspace_root))
        assert opened.status == created_workspace.status

    def test_preserves_folders(
        self, created_workspace: ProjectWorkspace,
        manager: WorkspaceManager, workspace_root: Path,
    ) -> None:
        opened = manager.open_workspace(str(workspace_root))
        assert len(opened.folders) == len(created_workspace.folders)

    def test_rejects_missing_root(self, manager: WorkspaceManager, tmp_path: Path) -> None:
        with pytest.raises(WorkspaceManagerError, match="does not exist"):
            manager.open_workspace(str(tmp_path / "nonexistent"))

    def test_rejects_missing_project_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(WorkspaceManagerError, match=_PROJECT_FILE):
            manager.open_workspace(str(empty_dir))

    def test_rejects_corrupted_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        ws_dir = tmp_path / "corrupted"
        ws_dir.mkdir()
        (ws_dir / _PROJECT_FILE).write_text("{bad json!!!", encoding="utf-8")
        with pytest.raises(WorkspaceManagerError, match="Failed to read"):
            manager.open_workspace(str(ws_dir))

    def test_rejects_invalid_metadata_in_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        ws_dir = tmp_path / "invalid_meta"
        ws_dir.mkdir()
        data = {"workspace_id": "x", "root_path": "x", "metadata": {"project_name": ""}}
        (ws_dir / _PROJECT_FILE).write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(WorkspaceManagerError, match="Invalid workspace data"):
            manager.open_workspace(str(ws_dir))

    def test_rejects_empty_path(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.open_workspace("")

    def test_rejects_whitespace_path(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.open_workspace("   ")


# -----------------------------------------------------------------------
# EXISTS tests
# -----------------------------------------------------------------------

class TestWorkspaceExists:
    """Tests for WorkspaceManager.workspace_exists."""

    def test_true_when_exists(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        assert manager.workspace_exists(str(workspace_root)) is True

    def test_false_when_missing_dir(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        assert manager.workspace_exists(str(tmp_path / "nope")) is False

    def test_false_when_missing_project_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert manager.workspace_exists(str(empty)) is False

    def test_rejects_empty_path(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.workspace_exists("")


# -----------------------------------------------------------------------
# DELETE tests
# -----------------------------------------------------------------------

class TestDeleteWorkspace:
    """Tests for WorkspaceManager.delete_workspace."""

    def test_deletes_workspace(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        assert workspace_root.is_dir()
        manager.delete_workspace(str(workspace_root))
        assert not workspace_root.exists()

    def test_recursive_deletion(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        # Add a nested file inside a sub-folder
        nested = workspace_root / "models" / "deep" / "file.txt"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("data", encoding="utf-8")
        manager.delete_workspace(str(workspace_root))
        assert not workspace_root.exists()

    def test_rejects_non_existent(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        with pytest.raises(WorkspaceManagerError, match="does not exist"):
            manager.delete_workspace(str(tmp_path / "ghost"))

    def test_rejects_empty_path(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.delete_workspace("")

    def test_workspace_exists_false_after_delete(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        manager.delete_workspace(str(workspace_root))
        assert manager.workspace_exists(str(workspace_root)) is False


# -----------------------------------------------------------------------
# VALIDATE tests
# -----------------------------------------------------------------------

class TestValidateWorkspace:
    """Tests for WorkspaceManager.validate_workspace."""

    def test_valid_workspace(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        assert manager.validate_workspace(str(workspace_root)) is True

    def test_invalid_missing_root(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        assert manager.validate_workspace(str(tmp_path / "nope")) is False

    def test_invalid_missing_project_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert manager.validate_workspace(str(empty)) is False

    def test_invalid_corrupted_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        ws = tmp_path / "broken"
        ws.mkdir()
        (ws / _PROJECT_FILE).write_text("not json", encoding="utf-8")
        assert manager.validate_workspace(str(ws)) is False

    def test_invalid_missing_folder(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        manager.create_workspace(str(workspace_root), metadata)
        # Remove one required folder
        import shutil
        shutil.rmtree(workspace_root / "models")
        assert manager.validate_workspace(str(workspace_root)) is False

    def test_invalid_bad_metadata_in_json(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        ws = tmp_path / "bad_meta"
        ws.mkdir()
        data = {"workspace_id": "w", "root_path": "r", "metadata": {}}
        (ws / _PROJECT_FILE).write_text(json.dumps(data), encoding="utf-8")
        assert manager.validate_workspace(str(ws)) is False

    def test_rejects_empty_path(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError, match="empty"):
            manager.validate_workspace("")


# -----------------------------------------------------------------------
# NON-MUTATION tests
# -----------------------------------------------------------------------

class TestNonMutation:
    """Ensure the manager never mutates input objects."""

    def test_metadata_not_mutated(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        original_name = metadata.project_name
        original_type = metadata.project_type
        original_ts = metadata.created_timestamp
        original_tags = list(metadata.tags)

        manager.create_workspace(str(workspace_root), metadata)

        assert metadata.project_name == original_name
        assert metadata.project_type == original_type
        assert metadata.created_timestamp == original_ts
        assert metadata.tags == original_tags

    def test_create_returns_new_object(
        self, manager: WorkspaceManager, tmp_path: Path, metadata: ProjectMetadata
    ) -> None:
        ws1 = manager.create_workspace(str(tmp_path / "ws1"), metadata)
        ws2 = manager.create_workspace(str(tmp_path / "ws2"), metadata)
        assert ws1 is not ws2
        assert ws1.workspace_id != ws2.workspace_id

    def test_open_returns_new_object(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        original = manager.create_workspace(str(workspace_root), metadata)
        opened = manager.open_workspace(str(workspace_root))
        assert original is not opened
        assert original.workspace_id == opened.workspace_id


# -----------------------------------------------------------------------
# ROUND-TRIP serialization tests
# -----------------------------------------------------------------------

class TestRoundTrip:
    """Verify create → open produces equivalent data."""

    def test_full_round_trip(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata
    ) -> None:
        created = manager.create_workspace(str(workspace_root), metadata)
        opened = manager.open_workspace(str(workspace_root))

        assert created.workspace_id == opened.workspace_id
        assert created.status == opened.status
        assert created.root_path == opened.root_path
        assert created.metadata.project_name == opened.metadata.project_name
        assert created.metadata.project_type == opened.metadata.project_type
        assert created.metadata.created_timestamp == opened.metadata.created_timestamp
        assert created.metadata.description == opened.metadata.description
        assert created.metadata.tags == opened.metadata.tags
        assert len(created.folders) == len(opened.folders)

    def test_json_file_round_trip(
        self, workspace_root: Path, created_workspace: ProjectWorkspace
    ) -> None:
        raw = (workspace_root / _PROJECT_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
        restored = ProjectWorkspace.model_validate(data)
        assert restored.workspace_id == created_workspace.workspace_id
        assert restored.metadata.project_name == created_workspace.metadata.project_name

    def test_round_trip_with_tags_and_description(
        self, manager: WorkspaceManager, tmp_path: Path
    ) -> None:
        meta = ProjectMetadata(
            project_name="Tagged Project",
            project_type=ProjectType.REGRESSION,
            created_timestamp="2025-12-01T00:00:00Z",
            description="A described project.",
            tags=["alpha", "beta", "gamma"],
        )
        root = tmp_path / "tagged_ws"
        created = manager.create_workspace(str(root), meta)
        opened = manager.open_workspace(str(root))
        assert opened.metadata.description == "A described project."
        assert opened.metadata.tags == ["alpha", "beta", "gamma"]


# -----------------------------------------------------------------------
# ERROR WRAPPING tests
# -----------------------------------------------------------------------

class TestErrorWrapping:
    """All filesystem errors wrapped in WorkspaceManagerError."""

    def test_workspace_manager_error_is_exception(self) -> None:
        assert issubclass(WorkspaceManagerError, Exception)

    def test_create_wraps_error_message(
        self, manager: WorkspaceManager, metadata: ProjectMetadata
    ) -> None:
        with pytest.raises(WorkspaceManagerError):
            manager.create_workspace("", metadata)

    def test_open_wraps_error_message(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError):
            manager.open_workspace("")

    def test_delete_wraps_error_message(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError):
            manager.delete_workspace("")

    def test_validate_wraps_error_message(self, manager: WorkspaceManager) -> None:
        with pytest.raises(WorkspaceManagerError):
            manager.validate_workspace("")


# -----------------------------------------------------------------------
# FILESYSTEM EDGE-CASE tests (for 100% coverage)
# -----------------------------------------------------------------------

class TestFilesystemEdgeCases:
    """Exercises hard-to-reach filesystem error branches."""

    def test_create_rejects_existing_file_at_root_path(
        self, manager: WorkspaceManager, tmp_path: Path, metadata: ProjectMetadata
    ) -> None:
        """If root_path resolves to a file (not a dir), reject."""
        file_path = tmp_path / "a_file"
        file_path.write_text("I am a file", encoding="utf-8")
        with pytest.raises(WorkspaceManagerError, match="not a directory"):
            manager.create_workspace(str(file_path), metadata)

    def test_create_wraps_mkdir_os_error(
        self, manager: WorkspaceManager, tmp_path: Path, metadata: ProjectMetadata, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate an OSError during root mkdir."""
        target = tmp_path / "fail_dir"

        original_mkdir = Path.mkdir

        def _failing_mkdir(self_path: Path, *args: object, **kwargs: object) -> None:
            if self_path == target:
                raise OSError("disk full")
            return original_mkdir(self_path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _failing_mkdir)
        with pytest.raises(WorkspaceManagerError, match="Failed to create workspace directory"):
            manager.create_workspace(str(target), metadata)

    def test_create_wraps_subfolder_os_error(
        self, manager: WorkspaceManager, tmp_path: Path, metadata: ProjectMetadata, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate an OSError during sub-folder mkdir."""
        target = tmp_path / "ws_subfail"
        call_count = 0
        original_mkdir = Path.mkdir

        def _failing_mkdir(self_path: Path, *args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            # Let root mkdir succeed (call 1), fail on first subfolder (call 2)
            if call_count >= 2:
                raise OSError("permission denied")
            return original_mkdir(self_path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", _failing_mkdir)
        with pytest.raises(WorkspaceManagerError, match="Failed to create folder"):
            manager.create_workspace(str(target), metadata)

    def test_create_wraps_write_failure(
        self, manager: WorkspaceManager, tmp_path: Path, metadata: ProjectMetadata, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate an OSError when writing project.json."""
        target = tmp_path / "ws_writefail"

        original_write = Path.write_text

        def _failing_write(self_path: Path, *args: object, **kwargs: object) -> None:
            if self_path.name == "project.json":
                raise OSError("read-only filesystem")
            return original_write(self_path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _failing_write)
        with pytest.raises(WorkspaceManagerError, match="Failed to write"):
            manager.create_workspace(str(target), metadata)

    def test_delete_wraps_rmtree_failure(
        self, manager: WorkspaceManager, workspace_root: Path, metadata: ProjectMetadata, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate an OSError during shutil.rmtree."""
        import shutil as _shutil

        manager.create_workspace(str(workspace_root), metadata)

        def _failing_rmtree(*args: object, **kwargs: object) -> None:
            raise OSError("device busy")

        monkeypatch.setattr(_shutil, "rmtree", _failing_rmtree)
        with pytest.raises(WorkspaceManagerError, match="Failed to delete"):
            manager.delete_workspace(str(workspace_root))

    def test_create_into_existing_empty_dir(
        self, manager: WorkspaceManager, tmp_path: Path, metadata: ProjectMetadata
    ) -> None:
        """Creating into a pre-existing empty dir (no project.json) should succeed."""
        ws_dir = tmp_path / "preexisting"
        ws_dir.mkdir()
        ws = manager.create_workspace(str(ws_dir), metadata)
        assert isinstance(ws, ProjectWorkspace)
        assert (ws_dir / "project.json").is_file()
