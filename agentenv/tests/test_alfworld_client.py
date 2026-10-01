from __future__ import annotations

import unittest
from unittest.mock import patch

from agentenv.envs.alfworld import AlfWorldEnvClient


class Response:
    def __init__(self, payload: dict, status_error: Exception | None = None) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> dict:
        return self.payload


class AlfWorldClientTests(unittest.TestCase):
    def test_index_maps_to_explicit_stable_task_id(self) -> None:
        task_id = "valid_seen:pick_and_place_simple/trial-1"
        reset_payload = {
            "task_id": task_id,
            "split": "valid_seen",
            "observation": "start",
            "available_actions": ["look"],
            "reward": 0.0,
            "done": False,
            "success": None,
            "info": {},
        }
        with patch(
            "agentenv.envs.alfworld.requests.post",
            side_effect=[
                Response({"session_id": "session-1"}),
                Response(reset_payload),
                Response({"closed": True, "already_closed": False}),
            ],
        ) as post:
            client = AlfWorldEnvClient("http://alfworld", [task_id], 42)
            self.assertEqual(len(client), 1)
            self.assertEqual(client.reset(0), reset_payload)
            self.assertEqual(client.close(), {"closed": True, "already_closed": False})

        self.assertEqual(
            post.call_args_list[1].kwargs["json"],
            {"session_id": "session-1", "task_id": task_id, "seed": 42},
        )
        self.assertEqual(post.call_args_list[2].kwargs["json"], {"session_id": "session-1"})

    def test_invalid_task_inputs_fail_before_reset_request(self) -> None:
        with patch(
            "agentenv.envs.alfworld.requests.post",
            return_value=Response({"session_id": "session-1"}),
        ):
            with self.assertRaises(ValueError):
                AlfWorldEnvClient("http://alfworld", [], 42)
            client = AlfWorldEnvClient("http://alfworld", ["train:task/trial"], 42)
            with self.assertRaises(TypeError):
                client.reset(-1)
            with self.assertRaises(IndexError):
                client.reset(1)


if __name__ == "__main__":
    unittest.main()
