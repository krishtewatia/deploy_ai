"""Pydantic v2 schemas for the Project Workspace module.

Defines ProjectStatus, ProjectType enums, and
WorkspaceFolder, ProjectMetadata, ProjectWorkspace models.

This module is SCHEMAS ONLY — no filesystem operations, no directory
creation, no file IO, no pathlib, no os.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Invalid characters for folder names and paths
# ---------------------------------------------------------------------------
_INVALID_CHARS_PATTERN = re.compile(r'[/\\:*?"<>|]')


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProjectStatus(str, Enum):
    """Lifecycle status of a project workspace."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    READ_ONLY = "READ_ONLY"


class ProjectType(str, Enum):
    """The problem category of the project."""

    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    GENERAL = "GENERAL"


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def _clean_string(field_name: str, v: Any) -> str:
    """Strip whitespace and reject empty strings."""
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = v.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return cleaned


def _validate_no_invalid_chars(field_name: str, v: str) -> str:
    """Reject strings containing invalid filesystem characters."""
    if _INVALID_CHARS_PATTERN.search(v):
        raise ValueError(
            f"{field_name} contains invalid characters "
            f"(must not contain / \\ : * ? \" < > |)"
        )
    return v


# ---------------------------------------------------------------------------
# WorkspaceFolder
# ---------------------------------------------------------------------------

class WorkspaceFolder(BaseModel):
    """A logical folder within a project workspace."""

    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(..., description="Logical name of the folder (e.g. 'models').")
    relative_path: str = Field(
        ...,
        description="Relative path of the folder within the workspace root.",
    )
    description: Optional[str] = Field(
        None, description="Optional description of the folder purpose."
    )

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, v: Any, info: Any) -> str:
        cleaned = _clean_string(info.field_name, v)
        return _validate_no_invalid_chars(info.field_name, cleaned)

    @field_validator("relative_path", mode="before")
    @classmethod
    def _validate_relative_path(cls, v: Any, info: Any) -> str:
        cleaned = _clean_string(info.field_name, v)
        return _validate_no_invalid_chars(info.field_name, cleaned)

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        return _clean_string(info.field_name, v)


# ---------------------------------------------------------------------------
# ProjectMetadata
# ---------------------------------------------------------------------------

class ProjectMetadata(BaseModel):
    """Immutable metadata associated with a project workspace."""

    model_config = ConfigDict(use_enum_values=True)

    project_name: str = Field(..., description="Human-readable project name.")
    project_type: ProjectType = Field(
        ..., description="The problem category of the project."
    )
    created_timestamp: str = Field(
        ..., description="ISO UTC timestamp when the project was created."
    )
    description: Optional[str] = Field(
        None, description="Optional description of the project."
    )
    tags: List[str] = Field(
        default_factory=list, description="Free-form tags for discovery."
    )

    @field_validator("project_name", "created_timestamp", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description(cls, v: Any, info: Any) -> Optional[str]:
        if v is None:
            return None
        return _clean_string(info.field_name, v)

    @field_validator("tags", mode="before")
    @classmethod
    def _validate_tags(cls, v: Any, info: Any) -> List[str]:
        if not isinstance(v, list):
            raise ValueError("tags must be a list")
        cleaned: List[str] = []
        for i, tag in enumerate(v):
            if not isinstance(tag, str):
                raise ValueError(f"tags[{i}] must be a string")
            stripped = tag.strip()
            if not stripped:
                raise ValueError(f"tags[{i}] cannot be empty or whitespace-only")
            cleaned.append(stripped)
        return cleaned


# ---------------------------------------------------------------------------
# Default workspace folders
# ---------------------------------------------------------------------------

DEFAULT_FOLDERS: List[WorkspaceFolder] = [
    WorkspaceFolder(
        name="datasets",
        relative_path="datasets",
        description="Uploaded and processed dataset files.",
    ),
    WorkspaceFolder(
        name="models",
        relative_path="models",
        description="Trained model artifacts and checkpoints.",
    ),
    WorkspaceFolder(
        name="reports",
        relative_path="reports",
        description="Generated execution and executive reports.",
    ),
    WorkspaceFolder(
        name="logs",
        relative_path="logs",
        description="Training and pipeline execution logs.",
    ),
    WorkspaceFolder(
        name="configs",
        relative_path="configs",
        description="Configuration and plan files.",
    ),
]


# ---------------------------------------------------------------------------
# ProjectWorkspace
# ---------------------------------------------------------------------------

class ProjectWorkspace(BaseModel):
    """Top-level schema representing a project workspace."""

    model_config = ConfigDict(use_enum_values=True)

    workspace_id: str = Field(..., description="Unique workspace identifier.")
    status: ProjectStatus = Field(
        default=ProjectStatus.ACTIVE,
        description="Lifecycle status of the workspace.",
    )
    metadata: ProjectMetadata = Field(
        ..., description="Immutable metadata associated with the project."
    )
    root_path: str = Field(
        ..., description="Logical root path of the workspace (no IO performed)."
    )
    folders: List[WorkspaceFolder] = Field(
        default_factory=lambda: [f.model_copy() for f in DEFAULT_FOLDERS],
        description="Logical folder structure within the workspace.",
    )

    @field_validator("workspace_id", "root_path", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)

    @model_validator(mode="after")
    def _validate_unique_folders(self) -> ProjectWorkspace:
        """Ensure no duplicate folder names or relative paths."""
        seen_names: set[str] = set()
        seen_paths: set[str] = set()
        for folder in self.folders:
            lower_name = folder.name.lower()
            if lower_name in seen_names:
                raise ValueError(
                    f"Duplicate folder name detected: '{folder.name}'"
                )
            seen_names.add(lower_name)

            lower_path = folder.relative_path.lower()
            if lower_path in seen_paths:
                raise ValueError(
                    f"Duplicate folder relative_path detected: '{folder.relative_path}'"
                )
            seen_paths.add(lower_path)
        return self
