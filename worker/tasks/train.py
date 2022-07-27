from celery import Task
from worker.models import train_model


class TrainingTask(Task):
    """"""

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        return train_model()
