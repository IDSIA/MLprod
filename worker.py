from celery import Celery

import os


worker = Celery(
    'tasks',
    backend=os.getenv("CELERY_BACKEND_URL"),
    broker=os.getenv("CELERY_BROKER_URL"),
)

worker.conf.update(
    result_expires=3600,
)

@worker.task
def add(x, y):
    return x + y

@worker.task
def mul(x, y):
    return x * y

if __name__ == '__main__':
    worker.start()
