"""Recent Projects Registry — global registry of recently opened workspaces.

Persists as ``recent_projects.json``.  Does NOT create workspaces, does NOT
modify workspace contents.  Only manages the recent-projects list.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.workspace.schemas import ProjectWorkspace

_REGISTRY_FILE = "recent_projects.json"
_MAX_RECENT = 50


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RecentProjectsRegistryError(Exception):
    """Raised when a registry operation fails."""


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _clean_string(field_name: str, v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = v.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty or whitespace-only")
    return cleaned


class RecentProjectEntry(BaseModel):
    """A single entry in the recent-projects registry."""

    model_config = ConfigDict(use_enum_values=True)

    project_id: str = Field(..., description="Unique project / workspace ID.")
    project_name: str = Field(..., description="Human-readable project name.")
    workspace_path: str = Field(..., description="Absolute path to workspace root.")
    last_opened_timestamp: str = Field(
        ..., description="ISO UTC timestamp of last open."
    )
    pinned: bool = Field(default=False, description="Whether this entry is pinned.")

    @field_validator("project_id", "project_name", "workspace_path", "last_opened_timestamp", mode="before")
    @classmethod
    def _validate_non_empty_strings(cls, v: Any, info: Any) -> str:
        return _clean_string(info.field_name, v)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RecentProjectsRegistry:
    """Manages a JSON-backed list of recently opened workspaces.

    Parameters
    ----------
    registry_path : str
        Path to the directory where ``recent_projects.json`` is stored.
    """

    def __init__(self, registry_path: str) -> None:
        if not isinstance(registry_path, str) or not registry_path.strip():
            raise RecentProjectsRegistryError(
                "registry_path cannot be empty or whitespace-only"
            )
        self._registry_dir = Path(registry_path).resolve()
        self._registry_file = self._registry_dir / _REGISTRY_FILE
        self._entries: List[RecentProjectEntry] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_workspace(self, workspace: ProjectWorkspace) -> None:
        """Register or update a workspace in the recent list.

        If the workspace already exists (by ``workspace_path``), its
        ``last_opened_timestamp`` is updated.  Otherwise a new entry is
        appended.  The registry is then trimmed to ``_MAX_RECENT``.
        """
        if not isinstance(workspace, ProjectWorkspace):
            raise RecentProjectsRegistryError(
                "workspace must be a ProjectWorkspace instance"
            )

        now = datetime.now(timezone.utc).isoformat()
        path_key = self._normalise_path(workspace.root_path)

        existing = self._find_by_path(path_key)
        if existing is not None:
            # Update timestamp in-place (replace entry to stay immutable)
            idx = self._entries.index(existing)
            self._entries[idx] = RecentProjectEntry(
                project_id=existing.project_id,
                project_name=existing.project_name,
                workspace_path=existing.workspace_path,
                last_opened_timestamp=now,
                pinned=existing.pinned,
            )
        else:
            entry = RecentProjectEntry(
                project_id=workspace.workspace_id,
                project_name=workspace.metadata.project_name,
                workspace_path=workspace.root_path,
                last_opened_timestamp=now,
                pinned=False,
            )
            self._entries.append(entry)

        self._enforce_max_size()
        self._save()

    def remove_workspace(self, path: str) -> None:
        """Remove a workspace from the registry by its path."""
        self._reject_empty("path", path)
        path_key = self._normalise_path(path)
        entry = self._find_by_path(path_key)
        if entry is None:
            raise RecentProjectsRegistryError(
                f"No registry entry for path '{path}'"
            )
        self._entries.remove(entry)
        self._save()

    def pin_workspace(self, path: str) -> None:
        """Pin a workspace so it always appears first and is never evicted."""
        self._reject_empty("path", path)
        path_key = self._normalise_path(path)
        entry = self._find_by_path(path_key)
        if entry is None:
            raise RecentProjectsRegistryError(
                f"No registry entry for path '{path}'"
            )
        idx = self._entries.index(entry)
        self._entries[idx] = RecentProjectEntry(
            project_id=entry.project_id,
            project_name=entry.project_name,
            workspace_path=entry.workspace_path,
            last_opened_timestamp=entry.last_opened_timestamp,
            pinned=True,
        )
        self._save()

    def unpin_workspace(self, path: str) -> None:
        """Remove the pin from a workspace."""
        self._reject_empty("path", path)
        path_key = self._normalise_path(path)
        entry = self._find_by_path(path_key)
        if entry is None:
            raise RecentProjectsRegistryError(
                f"No registry entry for path '{path}'"
            )
        idx = self._entries.index(entry)
        self._entries[idx] = RecentProjectEntry(
            project_id=entry.project_id,
            project_name=entry.project_name,
            workspace_path=entry.workspace_path,
            last_opened_timestamp=entry.last_opened_timestamp,
            pinned=False,
        )
        self._save()

    def list_projects(self) -> List[RecentProjectEntry]:
        """Return all entries sorted: pinned first, then by last_opened desc."""
        return sorted(
            self._entries,
            key=lambda e: (not e.pinned, e.last_opened_timestamp),
            reverse=False,
        )

    def clear(self) -> None:
        """Remove all entries from the registry."""
        self._entries.clear()
        self._save()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_empty(field_name: str, value: str) -> None:
        if not isinstance(value, str):
            raise RecentProjectsRegistryError(f"{field_name} must be a string")
        if not value.strip():
            raise RecentProjectsRegistryError(
                f"{field_name} cannot be empty or whitespace-only"
            )

    @staticmethod
    def _normalise_path(p: str) -> str:
        """Normalise a path for comparison."""
        return str(Path(p).resolve())

    def _find_by_path(self, normalised: str) -> Optional[RecentProjectEntry]:
        for entry in self._entries:
            if self._normalise_path(entry.workspace_path) == normalised:
                return entry
        return None

    def _enforce_max_size(self) -> None:
        """Trim to _MAX_RECENT, evicting oldest non-pinned entries first."""
        if len(self._entries) <= _MAX_RECENT:
            return

        pinned = [e for e in self._entries if e.pinned]
        non_pinned = [e for e in self._entries if not e.pinned]
        # Sort non-pinned by timestamp desc (newest first)
        non_pinned.sort(key=lambda e: e.last_opened_timestamp, reverse=True)

        remaining_slots = _MAX_RECENT - len(pinned)
        if remaining_slots < 0:
            remaining_slots = 0
        non_pinned = non_pinned[:remaining_slots]

        self._entries = pinned + non_pinned

    def _load(self) -> None:
        """Load entries from disk.  Missing file → empty list."""
        if not self._registry_file.is_file():
            self._entries = []
            return
        try:
            raw = self._registry_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, list):
                raise RecentProjectsRegistryError(
                    "Registry file must contain a JSON array"
                )
            self._entries = [
                RecentProjectEntry.model_validate(item) for item in data
            ]
        except (json.JSONDecodeError, OSError) as exc:
            raise RecentProjectsRegistryError(
                f"Failed to load registry: {exc}"
            ) from exc
        except Exception as exc:
            raise RecentProjectsRegistryError(
                f"Invalid registry data: {exc}"
            ) from exc

    def _save(self) -> None:
        """Atomically persist entries to disk."""
        import os

        self._registry_dir.mkdir(parents=True, exist_ok=True)
        payload = [e.model_dump() for e in self._entries]
        raw = json.dumps(payload, indent=2, ensure_ascii=False)

        # Atomic write: write to temp file then rename
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self._registry_dir), suffix=".tmp"
            )
            # Close the fd immediately — write_text will open its own handle.
            # On Windows, the file cannot be written while fd is open.
            os.close(fd)

            tmp_path = Path(tmp_name)
            try:
                tmp_path.write_text(raw, encoding="utf-8")
                tmp_path.replace(self._registry_file)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        except RecentProjectsRegistryError:
            raise
        except Exception as exc:
            raise RecentProjectsRegistryError(
                f"Failed to save registry: {exc}"
            ) from exc

