import sys
from celery import Task

import torch
import pandas as pd

from database.database import SessionLocal
from database import crud

from worker.celery import worker
from worker.models import Model, PreProcess

import logging
import os
import sys


mms_path: str='./models/mms.model'
skb_path: str='./models/skb.model'
model_path: str='./models/neuralnet.model'
best_model_path: str='./models/best_neuralnet.model'


class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()

        self.model: Model = None
        self.preprocess: PreProcess = None

    def __call__(self, *args, **kwargs):
        if not self.model:
            logging.info(f'sys.path:{sys.path}')
            logging.info(f'cwd: {os.getcwd()}')
            logging.info('Loading model')
            self.model: Model = torch.load(model_path)
            logging.info('Loading pre-process models')
            self.preprocess = PreProcess(mms_path, skb_path)
            logging.info('All models loaded')

        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    path=('worker.models', 'Model', 'PreProcess')
)
def inference(self, uid: int):
    try:
        db = SessionLocal()

        user = crud.get_user(db, uid)
        locs = crud.get_locations(db)
    
        df = pd.DataFrame([user.__dict__ | loc.__dict__ for loc in locs])

        logging.info("dataframe columns:", df.columns)

        x = df.values

        x = self.preprocess(x)
        return self.model(x)
    finally:
        db.close()
