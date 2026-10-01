"""Single-process ALFWorld TextWorld service with explicit session lifecycle."""

from __future__ import annotations

import copy
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from uuid import uuid4

from .manifest import ManifestError, TaskManifest, TaskRecord


class ServiceError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class TextWorldEnvironment(Protocol):
    def reset(self) -> tuple[Any, Mapping[str, Any]]: ...
    def step(self, actions: list[str]) -> tuple[Any, Any, Any, Mapping[str, Any]]: ...
    def close(self) -> Any: ...


EnvironmentFactory = Callable[[TaskRecord, int], TextWorldEnvironment]


@dataclass
class Session:
    session_id: str
    lock: threading.RLock = field(default_factory=threading.RLock)
    environment: TextWorldEnvironment | None = None
    task: TaskRecord | None = None
    observation: str | None = None
    available_actions: tuple[str, ...] = ()
    reward: float = 0.0
    done: bool = False
    success: bool | None = None
    info: dict[str, Any] = field(default_factory=dict)


class ALFWorldService:
    """Session registry for one Uvicorn worker; environment calls are serialized."""

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        environment_factory: EnvironmentFactory,
        *,
        alfworld_version: str = "0.5.0",
        verify_hashes: bool = False,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        if not self.data_root.is_dir():
            raise ManifestError(f"ALFWorld data root is not a directory: {self.data_root}")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = TaskManifest.load(self.manifest_path)
        if self.manifest.alfworld_version != alfworld_version:
            raise ManifestError(
                f"manifest ALFWorld version {self.manifest.alfworld_version} does not match {alfworld_version}"
            )
        self.manifest.verify_files(self.data_root, verify_hashes=verify_hashes)
        self.tasks = self.manifest.by_id()
        self.environment_factory = environment_factory
        self.alfworld_version = alfworld_version
        self._registry_lock = threading.RLock()
        self._environment_lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._closed_sessions: set[str] = set()

    def health(self) -> dict[str, Any]:
        with self._registry_lock:
            active_sessions = len(self._sessions)
        return {
            "status": "ok",
            "alfworld_version": self.alfworld_version,
            "manifest_sha256": self.manifest.sha256,
            "active_sessions": active_sessions,
        }

    def create(self) -> dict[str, str]:
        session_id = uuid4().hex
        with self._registry_lock:
            self._sessions[session_id] = Session(session_id=session_id)
        return {"session_id": session_id}

    def reset(self, session_id: str, task_id: str, seed: int) -> dict[str, Any]:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ServiceError(422, "invalid_seed", "seed must be a non-negative integer")
        task = self.tasks.get(task_id)
        if task is None:
            raise ServiceError(404, "unknown_task", f"unknown task_id: {task_id}")
        session = self._session(session_id)
        with session.lock, self._environment_lock:
            self._close_environment(session)
            environment = self.environment_factory(task, seed)
            try:
                observations, info = environment.reset()
                observation = str(_batch_item(observations, "observation"))
                actions = _actions(info)
            except BaseException:
                _best_effort_close(environment)
                raise
            session.environment = environment
            session.task = task
            session.observation = observation
            session.available_actions = actions
            session.reward = 0.0
            session.done = False
            session.success = None
            session.info = _safe_info(info)
            return self._state_payload(session)

    def observation(self, session_id: str) -> dict[str, Any]:
        session = self._session(session_id)
        with session.lock:
            self._require_reset(session)
            return self._state_payload(session)

    def available_actions(self, session_id: str) -> dict[str, list[str]]:
        session = self._session(session_id)
        with session.lock:
            self._require_reset(session)
            return {"available_actions": list(session.available_actions)}

    def step(self, session_id: str, action: str) -> dict[str, Any]:
        if not isinstance(action, str) or not action or action != action.strip():
            raise ServiceError(422, "invalid_action", "action must be stripped non-empty text")
        session = self._session(session_id)
        with session.lock, self._environment_lock:
            self._require_reset(session)
            if session.done:
                raise ServiceError(409, "session_done", "cannot step a completed environment")
            if action not in session.available_actions:
                raise ServiceError(422, "inadmissible_action", "action is not currently admissible")
            assert session.environment is not None
            observations, scores, dones, info = session.environment.step([action])
            session.observation = str(_batch_item(observations, "observation"))
            session.available_actions = _actions(info)
            session.reward = float(_batch_item(scores, "reward"))
            session.done = bool(_batch_item(dones, "done"))
            won = info.get("won")
            session.success = bool(_batch_item(won, "won")) if session.done and won is not None else None
            session.info = _safe_info(info)
            return self._state_payload(session)

    def close(self, session_id: str) -> dict[str, bool]:
        with self._registry_lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                if session_id in self._closed_sessions:
                    return {"closed": False, "already_closed": True}
                raise ServiceError(404, "unknown_session", f"unknown session_id: {session_id}")
            self._closed_sessions.add(session_id)
        with session.lock, self._environment_lock:
            self._close_environment(session)
        return {"closed": True, "already_closed": False}

    def close_all(self) -> None:
        with self._registry_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            try:
                self.close(session_id)
            except Exception:
                pass

    def _session(self, session_id: str) -> Session:
        if not isinstance(session_id, str) or not session_id:
            raise ServiceError(422, "invalid_session", "session_id must be non-empty text")
        with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ServiceError(404, "unknown_session", f"unknown session_id: {session_id}")
        return session

    @staticmethod
    def _require_reset(session: Session) -> None:
        if session.environment is None or session.task is None or session.observation is None:
            raise ServiceError(409, "session_not_reset", "reset must be called before this operation")

    @staticmethod
    def _close_environment(session: Session) -> None:
        if session.environment is not None:
            session.environment.close()
        session.environment = None
        session.task = None
        session.observation = None
        session.available_actions = ()
        session.reward = 0.0
        session.done = False
        session.success = None
        session.info = {}

    @staticmethod
    def _state_payload(session: Session) -> dict[str, Any]:
        assert session.task is not None and session.observation is not None
        return {
            "task_id": session.task.task_id,
            "split": session.task.split,
            "observation": session.observation,
            "available_actions": list(session.available_actions),
            "reward": session.reward,
            "done": session.done,
            "success": session.success,
            "info": copy.deepcopy(session.info),
        }


def official_environment_factory(data_root: Path, config_path: Path) -> EnvironmentFactory:
    """Create an ALFWorld 0.5.0 TextWorld environment for one manifest task."""

    import numpy as np
    import yaml
    from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    class SingleGameAlfredTWEnv(AlfredTWEnv):
        def __init__(self, task: TaskRecord, seed: int) -> None:
            self.config = copy.deepcopy(config)
            self.train_eval = {
                "train": "train",
                "valid_seen": "eval_in_distribution",
                "valid_unseen": "eval_out_of_distribution",
            }[task.split]
            self.game_files = [str((data_root / task.game_file).resolve())]
            self.num_games = 1
            self.config["general"]["random_seed"] = seed
            self.config["general"]["use_cuda"] = False
            self.config["env"]["domain_randomization"] = False

    def create(task: TaskRecord, seed: int) -> TextWorldEnvironment:
        random.seed(seed)
        np.random.seed(seed)
        return SingleGameAlfredTWEnv(task, seed).init_env(batch_size=1)

    return create


def _batch_item(value: Any, field: str) -> Any:
    try:
        if len(value) == 0:
            raise ServiceError(500, "invalid_environment_response", f"empty {field}")
        return value[0]
    except (KeyError, TypeError, IndexError) as error:
        raise ServiceError(500, "invalid_environment_response", f"invalid {field}") from error


def _actions(info: Mapping[str, Any]) -> tuple[str, ...]:
    actions = _batch_item(info.get("admissible_commands"), "admissible_commands")
    if not isinstance(actions, (list, tuple)) or not all(isinstance(action, str) and action for action in actions):
        raise ServiceError(500, "invalid_environment_response", "invalid admissible_commands")
    return tuple(dict.fromkeys(actions))


def _safe_info(info: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("won", "extra.gamefile"):
        if key in info:
            value = info[key]
            if hasattr(value, "tolist"):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)
            result[key] = value
    return result


def _best_effort_close(environment: TextWorldEnvironment) -> None:
    try:
        environment.close()
    except Exception:
        pass
