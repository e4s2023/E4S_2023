#!/usr/bin/env python

import os
from argparse import ArgumentParser

import uvicorn
from e4s2024 import rest_api


def _cli_parser() -> ArgumentParser:
    parser = ArgumentParser(description="REST API for face-swapping")
    parser.add_argument(
        "-p", "--port", default=int(os.environ.get("PORT", "3003")), type=int
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--data-dir", default=rest_api.DATA_DIR)
    return parser


if __name__ == "__main__":
    args = _cli_parser().parse_args()
    rest_api.DATA_DIR = args.data_dir

    from e4s2024.rest_api.routes import rest_api_app

    uvicorn.run(rest_api_app, host=args.host, port=args.port, forwarded_allow_ips="*")
