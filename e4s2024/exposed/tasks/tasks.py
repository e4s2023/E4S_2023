import os
from base64 import b64encode
from io import BytesIO
from typing import IO

from celery import Celery

from e4s2024 import exposed
from e4s2024.gradio_swap import swap_image, global_holder

broker_url = os.environ.get(
    "CELERY_BROKER_URL", os.environ.get("REDIS_URL", "redis://localhost:6379")
)
result_backend = os.environ.get(
    "CELERY_RESULT_BACKEND", os.environ.get("REDIS_URL", "redis://localhost:6379")
)
celery = Celery(__name__)
celery.conf.broker_url = broker_url
celery.conf.result_backend = result_backend
celery.conf.accept_content = ['pickle', 'json', 'msgpack', 'yaml']
celery.conf.worker_send_task_events = True


@celery.task(name="swap_image_task")
def swap_image_task(user_img_path: str, model_img_path: str) -> str:
    result = swap_image(
        global_holder["image"],
        user_img_path,
        model_img_path,
        exposed.DATA_DIR,
    )

    # Store the bytes on disk? - Or just send base64 to user, like below?
    iob: IO[bytes] = BytesIO()
    result.save(iob, format="png")
    iob.seek(0)
    output_url = "data:image/png;base64,{}".format(b64encode(iob.read()))

    return output_url


if __name__ == "__main__":
    celery.start()
