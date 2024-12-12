import traceback
from base64 import b64encode
from io import BytesIO
from typing import IO, Optional

from celery.result import AsyncResult
from pydantic import BaseModel
from fastapi import FastAPI, Response, status

from e4s2024 import rest_api, __version__
from e4s2024.gradio_swap import load_image_pipeline, swap_image, global_holder
from e4s2024.rest_api.helpers import url_to_path
from e4s2024.rest_api.tasks import swap_image_task


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


rest_api_app = FastAPI(
    openapi_url="/swap_docs/openapi.json",
    redoc_url="/swap_docs/redoc",
    swagger_ui_oauth2_redirect_url="/api/token",
)


@rest_api_app.get("/v1/swap/version")
def version() -> VersionResponse:
    return VersionResponse(version=__version__)


@rest_api_app.post("/v1/swap")
def swap(req: SwapRequest, res: Response) -> QueuedResponse | ErrorResponse:
    user_img_url = req.user_img_url  # user
    model_img_url = req.model_img_url  # model
    if user_img_url is None or model_img_url is None:
        res.status_code = status.HTTP_400_BAD_REQUEST
        return ErrorResponse(
            error="Missing parameter",
            error_description="Both `user_img_url` and `model_img_url` required",
            traceback=None,
        )

    if global_holder.get("image") is None:
        global_holder["image"] = load_image_pipeline()

    user_img_path = url_to_path(user_img_url)
    model_img_path = url_to_path(model_img_url)
    task = swap_image_task.delay(user_img_path, model_img_path)
    return QueuedResponse(task_id=task.id)


class TaskQueueStatusResponse(BaseModel):
    task_id: str
    task_status: str
    task_result: str


@rest_api_app.get("/v1/swap/{task_id}")
def get_status(task_id) -> TaskQueueStatusResponse:
    task_result = AsyncResult(task_id)
    if task_result.ready():
        return SwapResponse(output_url=task_result.get(), status="SUCCESS")
    else:
        return TaskQueueStatusResponse(
            task_id=task_id,
            task_status=task_result.status(),
            task_result=task_result.result,
        )
    #
    # try:
    #     result = swap_image(
    #         global_holder["image"],
    #         user_img_path,
    #         model_img_path,
    #         rest_api.DATA_DIR,
    #     )
    # except Exception as e:
    #     res.status_code = status.HTTP_400_BAD_REQUEST
    #     return ErrorResponse(
    #         error="Exception",
    #         error_description=str(e.args) if e.args else "{}".format(e),
    #         traceback=traceback.format_exc(),
    #     )
    # # Store the bytes on disk? - Or just send base64 to user, like below?
    # iob: IO[bytes] = BytesIO()
    # result.save(iob, format="png")
    # iob.seek(0)
    # output_url = "data:image/png;base64,{}".format(b64encode(iob.read()))
    #
    # # os.remove(user_img_path)
    # # os.remove(model_img_path)
    #
    # # TODO: queued rather than instant output
    # return SwapResponse(output_url=output_url, status="processed")
