from celery import Task

from database.database import SessionLocal
from database import crud

from worker.celery import worker
from worker.models import train_model

from datetime import datetime

import os
import logging


class TrainingTask(Task):
    """"""

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


@worker.task(
    ignore_results=True,
    bind=True,
    base=TrainingTask
)
def train(self):
    try:
        db = SessionLocal()

        X_tr, Y_tr = crud.get_dataset(db)

        path = os.path.join('.', 'models', f'model.{str(datetime.now())}')

        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        metrics = train_model(X_tr, Y_tr, path=path, metrics_list=metrics_list)

        crud.create_model(db, path, metrics)        

    finally:
        db.close()