#!/usr/bin/env python

from e4s2024.exposed.tasks.tasks import celery

if __name__ == "__main__":
    celery.start()
