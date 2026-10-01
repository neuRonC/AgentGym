from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agentenv_alfworld.manifest import ManifestError, TaskManifest
from agentenv_alfworld.service import ALFWorldService, ServiceError


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_manifest(root: Path) -> Path:
    logic_files = {}
    for relative, content in (("logic/alfred.pddl", "domain"), ("logic/alfred.twl2", "grammar")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logic_files[relative] = _sha256(path)
    tasks = []
    for split in ("train", "valid_seen", "valid_unseen"):
        relative = f"json_2.1.1/{split}/pick_and_place_simple/trial-{split}/game.tw-pddl"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(split, encoding="utf-8")
        tasks.append(
            {
                "task_id": f"{split}:pick_and_place_simple/trial-{split}",
                "split": split,
                "task_type": "pick_and_place_simple",
                "game_file": relative,
                "game_sha256": _sha256(path),
            }
        )
    tasks.sort(key=lambda task: task["task_id"])
    manifest = root / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "alfworld_version": "0.5.0",
                "data_format": "json_2.1.1",
                "logic_files": logic_files,
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


class FakeEnvironment:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.closed = False

    def reset(self) -> tuple[list[str], dict[str, Any]]:
        return [f"start:{self.task_id}"], {"admissible_commands": [["look", "inventory"]], "won": [False]}

    def step(self, actions: list[str]) -> tuple[list[str], list[float], list[bool], dict[str, Any]]:
        return [f"after:{actions[0]}:{self.task_id}"], [1.0], [True], {
            "admissible_commands": [["look"]],
            "won": [True],
        }

    def close(self) -> None:
        self.closed = True


class ManifestTests(unittest.TestCase):
    def test_manifest_tracks_official_splits_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _make_manifest(root)
            manifest = TaskManifest.load(manifest_path)
            manifest.verify_files(root, verify_hashes=True)
            self.assertEqual({task.split for task in manifest.tasks}, {"train", "valid_seen", "valid_unseen"})
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["tasks"][0]["game_file"] = "../escape/game.tw-pddl"
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "safe relative"):
                TaskManifest.load(manifest_path)


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest_path = _make_manifest(self.root)
        self.environments: list[FakeEnvironment] = []

        def factory(task: Any, seed: int) -> FakeEnvironment:
            self.assertEqual(seed, 42)
            environment = FakeEnvironment(task.task_id)
            self.environments.append(environment)
            return environment

        self.service = ALFWorldService(self.root, self.manifest_path, factory, verify_hashes=True)

    def tearDown(self) -> None:
        self.service.close_all()
        self.temporary.cleanup()

    def _task_id(self, split: str) -> str:
        return next(task.task_id for task in self.service.manifest.tasks if task.split == split)

    def test_lifecycle_reset_and_idempotent_close(self) -> None:
        session_id = self.service.create()["session_id"]
        task_id = self._task_id("valid_seen")
        reset = self.service.reset(session_id, task_id, 42)
        self.assertEqual(reset["available_actions"], ["look", "inventory"])
        first = self.environments[-1]
        self.service.reset(session_id, task_id, 42)
        self.assertTrue(first.closed)
        stepped = self.service.step(session_id, "look")
        self.assertEqual((stepped["reward"], stepped["done"], stepped["success"]), (1.0, True, True))
        self.assertEqual(self.service.close(session_id), {"closed": True, "already_closed": False})
        self.assertEqual(self.service.close(session_id), {"closed": False, "already_closed": True})

    def test_sessions_are_isolated_and_invalid_action_does_not_step(self) -> None:
        first_id = self.service.create()["session_id"]
        second_id = self.service.create()["session_id"]
        self.service.reset(first_id, self._task_id("train"), 42)
        self.service.reset(second_id, self._task_id("valid_unseen"), 42)
        with self.assertRaises(ServiceError) as context:
            self.service.step(first_id, "invented")
        self.assertEqual(context.exception.code, "inadmissible_action")
        self.service.step(first_id, "look")
        self.assertFalse(self.service.observation(second_id)["done"])

    def test_unknown_task_has_structured_error(self) -> None:
        session_id = self.service.create()["session_id"]
        with self.assertRaises(ServiceError) as context:
            self.service.reset(session_id, "valid_seen:not-real", 42)
        self.assertEqual((context.exception.status_code, context.exception.code), (404, "unknown_task"))


if __name__ == "__main__":
    unittest.main()
