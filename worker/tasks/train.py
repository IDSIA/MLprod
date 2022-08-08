from celery import Task

from database.database import SessionLocal
from database import crud

from worker.celery import worker
from worker.models import train_model
from worker.models.train import evaluate

from datetime import datetime

import os
import logging
import json


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
def training(self):
    try:
        task_id = str(self.request.id)
        db = SessionLocal()

        model = crud.get_best_model(db)

        # TODO: load existing model (maybe create an interm,ediate model for code sharing between the two tasks?)

        logging.info('All artifacts loaded')

        data = crud.create_dataset(db)

        # TODO: pass data to pandas (check columns) then split train/test data

        path = os.path.join('.', 'models', f'model.{str(datetime.now())}')

        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        metrics = train_model(X_tr, Y_tr, path=path, metrics_list=metrics_list)

        # TODO: test with old and new model

        old_model_metrics = evaluate(Y_ts, y_preds_prev, metrics_list)
        new_model_metrics = evaluate(Y_ts, y_preds_next, metrics_list)

        # TODO: save to disk and in database, update entry in model table

        crud.create_model(db, path, metrics)        

    finally:
        db.close()