from celery import Task

from worker.celery import worker
from models import DummyModel


class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model: DummyModel = DummyModel()

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    path=('models.dummy', 'DummyModel')
)
def predict(self, x):
    return self.model(x)
