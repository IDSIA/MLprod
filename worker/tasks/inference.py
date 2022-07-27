from celery import Task

import torch
import pandas as pd

from worker.celery import worker
from worker.models import Model, PreProcess
import logging


mms_path: str='./models/mms.model', 
skb_path: str='./models/skb.model',
model_path: str='./models/neuralnet.model',
best_model_path: str='./models/best_neuralnet.model',


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
def inference(self, user_data, locs):
    df = pd.DataFrame([user_data | loc for loc in locs])
    x = df.values
    x = self.preprocess(x)
    return self.model(x)
