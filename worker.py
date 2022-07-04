from celery import Celery

worker = Celery('tasks', broker='pyamqp://rabbitmq/', backend='redis://redis/')

worker.conf.update(
    result_expires=3600,
)

@worker.task
def add(x, y):
    return x + y

@worker.task
def mul(x, y):
    return x * y
