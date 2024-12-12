from typing import Optional

from pydantic import BaseModel


class SwapRequest(BaseModel):
    user_img_url: str
    model_img_url: str


class SwapResponse(BaseModel):
    output_url: str
    status: str


class QueuedResponse(BaseModel):
    task_id: str


class VersionResponse(BaseModel):
    version: str


class ErrorResponse(BaseModel):
    error: str
    error_description: str
    traceback: Optional[str]


class TaskQueueStatusResponse(BaseModel):
    task_id: str
    task_status: str
    task_result: str
