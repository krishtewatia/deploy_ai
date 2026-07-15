"""Comprehensive tests for the Project Workspace schemas (Stage 11A).

Covers:
- ProjectStatus and ProjectType enum values
- WorkspaceFolder validation (trimming, empty, invalid chars)
- ProjectMetadata validation (trimming, optional fields, tags)
- ProjectWorkspace defaults, uniqueness constraints, JSON serialization
"""

from __future__ import annotations

import json
import pytest
from pydantic import ValidationError

from backend.app.workspace.schemas import (
    DEFAULT_FOLDERS,
    ProjectMetadata,
    ProjectStatus,
    ProjectType,
    ProjectWorkspace,
    WorkspaceFolder,
)


# -----------------------------------------------------------------------
# Enum tests
# -----------------------------------------------------------------------

class TestProjectStatus:
    """Verify ProjectStatus enum members and string values."""

    def test_members(self) -> None:
        assert ProjectStatus.ACTIVE == "ACTIVE"
        assert ProjectStatus.ARCHIVED == "ARCHIVED"
        assert ProjectStatus.READ_ONLY == "READ_ONLY"

    def test_count(self) -> None:
        assert len(ProjectStatus) == 3


class TestProjectType:
    """Verify ProjectType enum members and string values."""

    def test_members(self) -> None:
        assert ProjectType.CLASSIFICATION == "CLASSIFICATION"
        assert ProjectType.REGRESSION == "REGRESSION"
        assert ProjectType.GENERAL == "GENERAL"

    def test_count(self) -> None:
        assert len(ProjectType) == 3


# -----------------------------------------------------------------------
# WorkspaceFolder tests
# -----------------------------------------------------------------------

class TestWorkspaceFolder:
    """WorkspaceFolder construction and validation."""

    def test_valid_creation(self) -> None:
        folder = WorkspaceFolder(
            name="models",
            relative_path="models",
            description="Model artifacts.",
        )
        assert folder.name == "models"
        assert folder.relative_path == "models"
        assert folder.description == "Model artifacts."

    def test_trims_name_and_path(self) -> None:
        folder = WorkspaceFolder(
            name="  models  ",
            relative_path="  models  ",
        )
        assert folder.name == "models"
        assert folder.relative_path == "models"

    def test_trims_description(self) -> None:
        folder = WorkspaceFolder(
            name="logs",
            relative_path="logs",
            description="  Execution logs  ",
        )
        assert folder.description == "Execution logs"

    def test_description_none_allowed(self) -> None:
        folder = WorkspaceFolder(name="data", relative_path="data")
        assert folder.description is None

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            WorkspaceFolder(name="", relative_path="some_path")

    def test_rejects_whitespace_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            WorkspaceFolder(name="   ", relative_path="some_path")

    def test_rejects_empty_relative_path(self) -> None:
        with pytest.raises(ValidationError, match="relative_path"):
            WorkspaceFolder(name="models", relative_path="")

    def test_rejects_whitespace_relative_path(self) -> None:
        with pytest.raises(ValidationError, match="relative_path"):
            WorkspaceFolder(name="models", relative_path="   ")

    def test_rejects_empty_description(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            WorkspaceFolder(name="models", relative_path="models", description="")

    def test_rejects_whitespace_description(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            WorkspaceFolder(name="m", relative_path="m", description="   ")

    @pytest.mark.parametrize("char", ["/", "\\", ":", "*", "?", '"', "<", ">", "|"])
    def test_rejects_invalid_chars_in_name(self, char: str) -> None:
        with pytest.raises(ValidationError, match="invalid characters"):
            WorkspaceFolder(name=f"my{char}folder", relative_path="safe")

    @pytest.mark.parametrize("char", ["/", "\\", ":", "*", "?", '"', "<", ">", "|"])
    def test_rejects_invalid_chars_in_relative_path(self, char: str) -> None:
        with pytest.raises(ValidationError, match="invalid characters"):
            WorkspaceFolder(name="safe", relative_path=f"my{char}path")

    def test_rejects_non_string_name(self) -> None:
        with pytest.raises(ValidationError, match="name"):
            WorkspaceFolder(name=123, relative_path="safe")  # type: ignore[arg-type]

    def test_rejects_non_string_relative_path(self) -> None:
        with pytest.raises(ValidationError, match="relative_path"):
            WorkspaceFolder(name="safe", relative_path=123)  # type: ignore[arg-type]

    def test_json_round_trip(self) -> None:
        folder = WorkspaceFolder(
            name="outputs", relative_path="outputs", description="Results."
        )
        data = json.loads(folder.model_dump_json())
        assert data["name"] == "outputs"
        assert data["relative_path"] == "outputs"
        assert data["description"] == "Results."


# -----------------------------------------------------------------------
# ProjectMetadata tests
# -----------------------------------------------------------------------

class TestProjectMetadata:
    """ProjectMetadata construction and validation."""

    def _make_metadata(self, **overrides: object) -> ProjectMetadata:
        defaults = dict(
            project_name="My Project",
            project_type=ProjectType.CLASSIFICATION,
            created_timestamp="2025-01-01T00:00:00Z",
        )
        defaults.update(overrides)
        return ProjectMetadata(**defaults)  # type: ignore[arg-type]

    def test_valid_creation(self) -> None:
        m = self._make_metadata()
        assert m.project_name == "My Project"
        assert m.project_type == "CLASSIFICATION"
        assert m.created_timestamp == "2025-01-01T00:00:00Z"
        assert m.description is None
        assert m.tags == []

    def test_trims_project_name(self) -> None:
        m = self._make_metadata(project_name="  Trimmed  ")
        assert m.project_name == "Trimmed"

    def test_trims_created_timestamp(self) -> None:
        m = self._make_metadata(created_timestamp="  2025-06-01T12:00:00Z  ")
        assert m.created_timestamp == "2025-06-01T12:00:00Z"

    def test_trims_description(self) -> None:
        m = self._make_metadata(description="  Some desc  ")
        assert m.description == "Some desc"

    def test_description_none_allowed(self) -> None:
        m = self._make_metadata(description=None)
        assert m.description is None

    def test_rejects_empty_project_name(self) -> None:
        with pytest.raises(ValidationError, match="project_name"):
            self._make_metadata(project_name="")

    def test_rejects_whitespace_project_name(self) -> None:
        with pytest.raises(ValidationError, match="project_name"):
            self._make_metadata(project_name="   ")

    def test_rejects_empty_created_timestamp(self) -> None:
        with pytest.raises(ValidationError, match="created_timestamp"):
            self._make_metadata(created_timestamp="")

    def test_rejects_empty_description(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            self._make_metadata(description="")

    def test_rejects_non_string_project_name(self) -> None:
        with pytest.raises(ValidationError, match="project_name"):
            self._make_metadata(project_name=42)

    def test_valid_tags(self) -> None:
        m = self._make_metadata(tags=["ml", "test"])
        assert m.tags == ["ml", "test"]

    def test_trims_tags(self) -> None:
        m = self._make_metadata(tags=["  ml  ", "  test  "])
        assert m.tags == ["ml", "test"]

    def test_rejects_empty_tag(self) -> None:
        with pytest.raises(ValidationError, match="tags"):
            self._make_metadata(tags=["ok", ""])

    def test_rejects_whitespace_tag(self) -> None:
        with pytest.raises(ValidationError, match="tags"):
            self._make_metadata(tags=["ok", "   "])

    def test_rejects_non_string_tag(self) -> None:
        with pytest.raises(ValidationError, match="tags"):
            self._make_metadata(tags=["ok", 123])

    def test_rejects_non_list_tags(self) -> None:
        with pytest.raises(ValidationError, match="tags"):
            self._make_metadata(tags="not_a_list")

    def test_all_project_types(self) -> None:
        for pt in ProjectType:
            m = self._make_metadata(project_type=pt)
            assert m.project_type == pt.value

    def test_json_round_trip(self) -> None:
        m = self._make_metadata(tags=["alpha", "beta"], description="Desc")
        data = json.loads(m.model_dump_json())
        assert data["project_name"] == "My Project"
        assert data["project_type"] == "CLASSIFICATION"
        assert data["tags"] == ["alpha", "beta"]
        assert data["description"] == "Desc"


# -----------------------------------------------------------------------
# Default folders tests
# -----------------------------------------------------------------------

class TestDefaultFolders:
    """Verify the DEFAULT_FOLDERS constant."""

    def test_count(self) -> None:
        assert len(DEFAULT_FOLDERS) == 5

    def test_names(self) -> None:
        names = [f.name for f in DEFAULT_FOLDERS]
        assert "datasets" in names
        assert "models" in names
        assert "reports" in names
        assert "logs" in names
        assert "configs" in names

    def test_all_have_descriptions(self) -> None:
        for f in DEFAULT_FOLDERS:
            assert f.description is not None
            assert len(f.description) > 0


# -----------------------------------------------------------------------
# ProjectWorkspace tests
# -----------------------------------------------------------------------

class TestProjectWorkspace:
    """ProjectWorkspace construction, defaults, and uniqueness."""

    def _make_metadata(self, **overrides: object) -> ProjectMetadata:
        defaults = dict(
            project_name="Test Project",
            project_type=ProjectType.REGRESSION,
            created_timestamp="2025-06-15T10:00:00Z",
        )
        defaults.update(overrides)
        return ProjectMetadata(**defaults)  # type: ignore[arg-type]

    def _make_workspace(self, **overrides: object) -> ProjectWorkspace:
        defaults: dict[str, object] = dict(
            workspace_id="ws-001",
            metadata=self._make_metadata(),
            root_path="project_root",
        )
        defaults.update(overrides)
        return ProjectWorkspace(**defaults)  # type: ignore[arg-type]

    def test_valid_creation_with_defaults(self) -> None:
        ws = self._make_workspace()
        assert ws.workspace_id == "ws-001"
        assert ws.status == "ACTIVE"
        assert ws.root_path == "project_root"
        assert len(ws.folders) == 5  # default folders

    def test_default_status_is_active(self) -> None:
        ws = self._make_workspace()
        assert ws.status == "ACTIVE"

    def test_explicit_status(self) -> None:
        ws = self._make_workspace(status=ProjectStatus.ARCHIVED)
        assert ws.status == "ARCHIVED"

    def test_all_statuses(self) -> None:
        for status in ProjectStatus:
            ws = self._make_workspace(status=status)
            assert ws.status == status.value

    def test_trims_workspace_id(self) -> None:
        ws = self._make_workspace(workspace_id="  ws-001  ")
        assert ws.workspace_id == "ws-001"

    def test_trims_root_path(self) -> None:
        ws = self._make_workspace(root_path="  root  ")
        assert ws.root_path == "root"

    def test_rejects_empty_workspace_id(self) -> None:
        with pytest.raises(ValidationError, match="workspace_id"):
            self._make_workspace(workspace_id="")

    def test_rejects_whitespace_workspace_id(self) -> None:
        with pytest.raises(ValidationError, match="workspace_id"):
            self._make_workspace(workspace_id="   ")

    def test_rejects_empty_root_path(self) -> None:
        with pytest.raises(ValidationError, match="root_path"):
            self._make_workspace(root_path="")

    def test_rejects_non_string_workspace_id(self) -> None:
        with pytest.raises(ValidationError, match="workspace_id"):
            self._make_workspace(workspace_id=999)

    def test_custom_folders(self) -> None:
        custom = [
            WorkspaceFolder(name="src", relative_path="src"),
            WorkspaceFolder(name="bin", relative_path="bin"),
        ]
        ws = self._make_workspace(folders=custom)
        assert len(ws.folders) == 2
        assert ws.folders[0].name == "src"
        assert ws.folders[1].name == "bin"

    def test_empty_folders_list_allowed(self) -> None:
        ws = self._make_workspace(folders=[])
        assert ws.folders == []

    def test_rejects_duplicate_folder_names(self) -> None:
        dups = [
            WorkspaceFolder(name="data", relative_path="data1"),
            WorkspaceFolder(name="data", relative_path="data2"),
        ]
        with pytest.raises(ValidationError, match="Duplicate folder name"):
            self._make_workspace(folders=dups)

    def test_rejects_duplicate_folder_names_case_insensitive(self) -> None:
        dups = [
            WorkspaceFolder(name="Data", relative_path="path1"),
            WorkspaceFolder(name="data", relative_path="path2"),
        ]
        with pytest.raises(ValidationError, match="Duplicate folder name"):
            self._make_workspace(folders=dups)

    def test_rejects_duplicate_folder_paths(self) -> None:
        dups = [
            WorkspaceFolder(name="alpha", relative_path="shared"),
            WorkspaceFolder(name="beta", relative_path="shared"),
        ]
        with pytest.raises(ValidationError, match="Duplicate folder relative_path"):
            self._make_workspace(folders=dups)

    def test_rejects_duplicate_folder_paths_case_insensitive(self) -> None:
        dups = [
            WorkspaceFolder(name="alpha", relative_path="Shared"),
            WorkspaceFolder(name="beta", relative_path="shared"),
        ]
        with pytest.raises(ValidationError, match="Duplicate folder relative_path"):
            self._make_workspace(folders=dups)

    def test_default_folders_are_independent_copies(self) -> None:
        """Each workspace gets its own copy of default folders."""
        ws1 = self._make_workspace()
        ws2 = self._make_workspace()
        assert ws1.folders is not ws2.folders
        assert ws1.folders[0] is not ws2.folders[0]

    def test_json_round_trip(self) -> None:
        ws = self._make_workspace()
        data = json.loads(ws.model_dump_json())
        assert data["workspace_id"] == "ws-001"
        assert data["status"] == "ACTIVE"
        assert data["root_path"] == "project_root"
        assert data["metadata"]["project_name"] == "Test Project"
        assert len(data["folders"]) == 5

    def test_json_deserialization(self) -> None:
        ws = self._make_workspace()
        raw = ws.model_dump_json()
        restored = ProjectWorkspace.model_validate_json(raw)
        assert restored.workspace_id == ws.workspace_id
        assert restored.status == ws.status
        assert restored.root_path == ws.root_path
        assert len(restored.folders) == len(ws.folders)

    def test_full_json_round_trip_custom(self) -> None:
        ws = self._make_workspace(
            status=ProjectStatus.READ_ONLY,
            folders=[WorkspaceFolder(name="custom", relative_path="custom_dir")],
        )
        raw = ws.model_dump_json()
        restored = ProjectWorkspace.model_validate_json(raw)
        assert restored.status == "READ_ONLY"
        assert len(restored.folders) == 1
        assert restored.folders[0].name == "custom"
