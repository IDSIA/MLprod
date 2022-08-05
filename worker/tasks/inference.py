from celery import Task

import torch
import pandas as pd

from database.database import SessionLocal
from database import crud

from worker.celery import worker
from worker.models import Model, PreProcess

import json
import logging
import os


class PredictTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()

        self.model: Model = None
        self.preprocess: PreProcess = None
        self.metadata: dict = None

        self.path_metadata: str=None
        self.path_mms: str=None
        self.path_skb: str=None
        self.path_model: str=None
        self.path: str=None
        self.set_path_to_use('./models/')

    def set_path_to_use(self, path: str) -> None:
        if self.path == path:
            return

        logging.info(f'setting new path to {path}')
        self.path = path
        self.path_metadata = str(os.path.join(self.path, 'metadata.json'))
        self.path_mms = str(os.path.join(self.path, 'mms.model'))
        self.path_skb = str(os.path.join(self.path, 'skb.model'))
        self.path_model = str(os.path.join(self.path, 'neuralnet.model'))
        self.model = None

        self.load_model()

    def load_model(self) -> None:
        logging.info(f'Load metadata from {self.path_metadata}')
        with open(self.path_metadata, 'r') as f:
            self.metadata = json.load(f)
        
        logging.info(f'Loading model from {self.path_model}')
        nn_state_dict = torch.load(self.path_model)
        self.model = Model(self.metadata['x_output'])
        self.model.load_state_dict(nn_state_dict)

        logging.info(f'Loading pre-process models from {self.path_mms} {self.path_skb}')
        self.preprocess = PreProcess(self.path_mms, self.path_skb)

        logging.info('All artifacts loaded')

    def __call__(self, *args, **kwargs):
        if self.model is None:
            self.load_model()

        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
)
def inference(self, user_id: int):
    try:
        db = SessionLocal()

        db_model = crud.get_best_model(db)
        self.set_path_to_use(db_model.path)

        user = crud.get_user(db, user_id)
        locs = crud.get_locations(db)

        locs_id = [loc.id for loc in locs]

        df = pd.DataFrame([user.__dict__ | loc.__dict__ for loc in locs], columns=self.metadata['features'])

        x = df.values
        x = self.preprocess(x)

        x = torch.FloatTensor(x)

        out = self.model(x)
        score = out.detach().numpy().astype('float')

        df['score'] = score
        df['user_id'] = user_id
        df['location_id'] = locs_id
        df['task_id'] = str(self.request.id)

        # save task id, user_id, and scores to database
        crud.create_results(db, df)

    finally:
        db.close()
