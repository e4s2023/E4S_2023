import os
from base64 import standard_b64decode
from random import randint
from urllib.parse import urlparse

import requests

from e4s2024 import exposed


def get_base64_part_decoded(url: str) -> bytes:
    return standard_b64decode(url[url.find(",") + 1 :])


def make_filepath(ext: str) -> str:
    return os.path.join(
        exposed.DATA_DIR, "{}{}{}".format(randint(20, 200), os.path.extsep, ext)
    )


def url_to_path(url: str) -> str:
    """Downloads URL or base64 and returns the location it downloaded to"""
    if url.startswith("data:image/"):
        url_part = url[len("data:image/") + 1]
        file_content = get_base64_part_decoded(url)
        if url_part.startswith("png;base64,") or url.startswith(
            "data:image/png;base64,"
        ):
            filepath = make_filepath("png")
        elif url_part.startswith("webm;base64,"):
            filepath = make_filepath("webm")
        elif url_part.startswith(("jpeg;base64,", "jpg;base64,")):
            filepath = make_filepath("jpg")
        else:
            raise NotImplementedError("{}...".format(url_part[:10]))

        with open(filepath, "wb") as f:
            f.write(file_content)
        return filepath
    elif url.startswith(("http://", "https://")):
        parse = urlparse(url)
        name = parse.path.replace("/", "__")
        filepath = os.path.join(exposed.DATA_DIR, name)
        # Cache check
        if not os.path.isfile(filepath):
            content = requests.get(url).content
            with open(filepath, "wb") as f:
                f.write(content)
        return filepath
    else:
        raise TypeError("Unaccepted url format")
