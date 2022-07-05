from worker.celery import worker


@worker.task
def add(x, y):
    return x + y


@worker.task
def mul(x, y):
    return x * y
