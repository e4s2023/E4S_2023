import os
from argparse import ArgumentParser

import uvicorn


DATA_DIR: str = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))


def _cli_parser() -> ArgumentParser:
    parser = ArgumentParser(description="REST API for face-swapping")
    parser.add_argument(
        "-p", "--port", default=int(os.environ.get("PORT", "3003")), type=int
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--data-dir", default=DATA_DIR)
    return parser


if __name__ == "__main__":
    from rest_api.routes import rest_api_app

    global_holder = {}
    args = _cli_parser()
    global DATA_DIR
    DATA_DIR = args.data_dir
    uvicorn.run(rest_api_app, host=args.host, port=args.port, forwarded_allow_ips="*")
