from celery import Task

import torch
import pandas as pd

from database.database import SessionLocal
from database import crud

from worker.celery import worker
from worker.models import Model, PreProcess

import json
import logging


metadata_path: str='./models/metadata.json'
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
        self.metadata: dict = None

    def __call__(self, *args, **kwargs):
        if not self.model:
            logging.info('Load metadata')
            
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            logging.info('Loading model')
            nn_state_dict = torch.load(model_path)
            self.model = Model(self.metadata['x_output'])
            self.model.load_state_dict(nn_state_dict)

            logging.info('Loading pre-process models')
            self.preprocess = PreProcess(mms_path, skb_path)

            logging.info('All artifacts loaded')

        return self.run(*args, **kwargs)


@worker.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
)
def inference(self, user_id: int):
    try:
        db = SessionLocal()

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
