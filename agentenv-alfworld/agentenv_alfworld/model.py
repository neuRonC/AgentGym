"""ALFWorld HTTP request models."""

from pydantic import BaseModel


class ResetRequestBody(BaseModel):
    session_id: str
    task_id: str
    seed: int


class StepRequestBody(BaseModel):
    session_id: str
    action: str


class SessionRequestBody(BaseModel):
    session_id: str

__all__ = ["ResetRequestBody", "SessionRequestBody", "StepRequestBody"]
