"""Validation for deterministic ALFWorld TextWorld task manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

MANIFEST_VERSION = 1
DATA_FORMAT = "json_2.1.1"
OFFICIAL_SPLITS = ("train", "valid_seen", "valid_unseen")
LOGIC_FILES = ("logic/alfred.pddl", "logic/alfred.twl2")


class ManifestError(ValueError):
    """The data tree or task manifest violates the service contract."""


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    split: str
    task_type: str
    game_file: str
    game_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> "TaskRecord":
        if not isinstance(value, Mapping):
            raise ManifestError("task entry must be an object")
        expected = {"task_id", "split", "task_type", "game_file", "game_sha256"}
        if set(value) != expected:
            raise ManifestError("task entry fields do not match manifest_version=1")
        record = cls(**{field: value[field] for field in expected})
        record.validate()
        return record

    def validate(self) -> None:
        if self.split not in OFFICIAL_SPLITS:
            raise ManifestError(f"unsupported official split: {self.split!r}")
        if not isinstance(self.task_type, str) or not self.task_type:
            raise ManifestError("task_type must be non-empty text")
        if not isinstance(self.game_file, str):
            raise ManifestError("game_file must be text")
        path = PurePosixPath(self.game_file)
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(f"game_file must be a safe relative path: {self.game_file!r}")
        if path.parts[:2] != (DATA_FORMAT, self.split) or path.name != "game.tw-pddl":
            raise ManifestError(f"game_file does not belong to split {self.split}: {self.game_file!r}")
        relative_parent = PurePosixPath(*path.parts[2:-1]).as_posix()
        expected_id = f"{self.split}:{relative_parent}"
        if self.task_id != expected_id:
            raise ManifestError(f"task_id {self.task_id!r} does not match {expected_id!r}")
        if not _is_digest(self.game_sha256):
            raise ManifestError("game_sha256 must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class TaskManifest:
    manifest_version: int
    alfworld_version: str
    data_format: str
    logic_files: Mapping[str, str]
    tasks: tuple[TaskRecord, ...]
    sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "TaskManifest":
        source = Path(path)
        raw = source.read_bytes()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ManifestError(f"cannot parse task manifest {source}: {error}") from error
        expected = {"manifest_version", "alfworld_version", "data_format", "logic_files", "tasks"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ManifestError("manifest fields do not match manifest_version=1")
        logic_files = value["logic_files"]
        if not isinstance(logic_files, Mapping) or set(logic_files) != set(LOGIC_FILES):
            raise ManifestError("logic_files must contain exactly the TextWorld logic inputs")
        if any(not _is_digest(digest) for digest in logic_files.values()):
            raise ManifestError("logic file hashes must be lowercase SHA256 digests")
        tasks = value["tasks"]
        if not isinstance(tasks, list):
            raise ManifestError("tasks must be a list")
        manifest = cls(
            manifest_version=value["manifest_version"],
            alfworld_version=value["alfworld_version"],
            data_format=value["data_format"],
            logic_files=dict(sorted(logic_files.items())),
            tasks=tuple(TaskRecord.from_mapping(task) for task in tasks),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.manifest_version != MANIFEST_VERSION:
            raise ManifestError(f"unsupported manifest_version: {self.manifest_version!r}")
        if self.alfworld_version != "0.5.0":
            raise ManifestError("this service requires alfworld_version=0.5.0")
        if self.data_format != DATA_FORMAT:
            raise ManifestError(f"unsupported data_format: {self.data_format!r}")
        task_ids = [task.task_id for task in self.tasks]
        if not task_ids:
            raise ManifestError("manifest contains no TextWorld tasks")
        if task_ids != sorted(task_ids):
            raise ManifestError("tasks must be sorted by task_id")
        if len(task_ids) != len(set(task_ids)):
            raise ManifestError("task_id values must be unique")
        missing = set(OFFICIAL_SPLITS) - {task.split for task in self.tasks}
        if missing:
            raise ManifestError(f"manifest is missing official splits: {', '.join(sorted(missing))}")

    def by_id(self) -> dict[str, TaskRecord]:
        return {task.task_id: task for task in self.tasks}

    def verify_files(self, data_root: str | Path, *, verify_hashes: bool = False) -> None:
        root = Path(data_root).resolve()
        for relative, expected_hash in self.logic_files.items():
            _verify_file(root, relative, expected_hash, verify_hashes)
        for task in self.tasks:
            _verify_file(root, task.game_file, task.game_sha256, verify_hashes)


def _verify_file(root: Path, relative: str, expected_hash: str, verify_hashes: bool) -> None:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ManifestError(f"manifest path escapes data root: {relative}")
    if not path.is_file():
        raise ManifestError(f"manifest file is missing: {path}")
    if verify_hashes and _sha256(path) != expected_hash:
        raise ManifestError(f"manifest hash mismatch: {relative}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
