# Agent Environments - ALFWorld TextWorld

This package exposes ALFWorld 0.5.0 TextWorld through a small HTTP service. The
service is intentionally single-process and single-worker: sessions live in
memory, while environment operations are serialized because the upstream
TextWorld stack mutates process-global state.

## Data and task identity

The service never downloads data. Prepare data separately with the official
`alfworld-download --data-dir <path>` command and provide both variables:

```sh
export ALFWORLD_DATA=/absolute/path/to/alfworld-data
export ALFWORLD_TASK_MANIFEST=/absolute/path/to/tasks.v1.json
```

The manifest must identify ALFWorld 0.5.0 tasks from the official `train`,
`valid_seen`, and `valid_unseen` directories. Requests select a task by stable
`task_id`; the legacy AgentGym mapping files and `valid_train` directory are not
used as official split labels.

`ALFWORLD_CONFIG` may override the packaged TextWorld config.
`ALFWORLD_VERIFY_HASHES=1` enables full input hashing at service startup.

## Install and launch

Use Python 3.9. Reproducible consumers should resolve and lock dependencies in
their deployment repository, install this package non-editably, and mount the
data and manifest read-only.

```sh
python -m pip install .
alfworld --host 0.0.0.0 --port 36002
```

The launcher always uses one Uvicorn worker.

## HTTP lifecycle

- `GET /health`
- `POST /create`
- `POST /reset` with `session_id`, `task_id`, and `seed`
- `GET /observation?session_id=...`
- `GET /available_actions?session_id=...`
- `POST /step` with `session_id` and an exact admissible TextWorld action
- `POST /close` with `session_id`

Errors use non-2xx status codes and structured `error.code` values. Repeating
`close` for the same known session succeeds with `already_closed: true`.

## Project modification notice

This directory retains the repository's MIT license and copyright notice. The
`ptct-alfworld-textworld` branch replaces the original ALFWorld 0.3.3 wrapper
with a manifest-addressed ALFWorld 0.5.0 lifecycle service and aligns the
unified AgentGym client with that API. Other AgentGym environments are outside
this change.
