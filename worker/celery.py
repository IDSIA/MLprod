from celery import Celery

import os


worker = Celery(
    'tasks',
    backend=os.getenv('CELERY_BACKEND_URL'),
    broker=os.getenv('CELERY_BROKER_URL'),
    include=[
        'worker.tasks',
    ]
)

worker.config_from_object('worker.celeryconfig')


if __name__ == '__main__':
    worker.start()
