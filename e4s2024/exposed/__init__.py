#!/usr/bin/env python

import os

DATA_DIR: str = os.environ.get("DATA_DIR", os.path.join(os.getcwd(), "data"))

__all__ = ["DATA_DIR"]
