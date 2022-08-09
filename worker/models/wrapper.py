from .model import Model
from .preprocess import PreProcess

import numpy as np

import logging
import torch
import json
import os

FILE_METADATA: str = 'metadata.json'
FILE_MMS: str = 'mms.model'
FILE_SKB: str = 'skb.model'
FILE_MODEL: str = 'neuralnet.model'


class InferenceModel:

    def __init__(self, path: str='./models/') -> None:
        super().__init__()

        self.path: str = path

        logging.info(f'loading model from path {self.path}')

        self.path_metadata: str = str(os.path.join(self.path, FILE_METADATA))
        self.path_mms: str = str(os.path.join(self.path, FILE_MMS))
        self.path_skb: str = str(os.path.join(self.path, FILE_SKB))
        self.path_model: str = str(os.path.join(self.path, FILE_MODEL))
        
        logging.info(f'Load metadata from {self.path_metadata}')

        with open(self.path_metadata, 'r') as f:
            self.metadata: dict = json.load(f)
        
        logging.info(f'Loading model from {self.path_model}')

        nn_state_dict = torch.load(self.path_model)
        self.model: Model = Model(self.metadata['x_output'])
        self.model.load_state_dict(nn_state_dict)

        logging.info(f'Loading pre-process models from {self.path_mms} {self.path_skb}')

        self.preprocess: PreProcess = PreProcess(self.path_mms, self.path_skb)

        logging.info('All artifacts loaded')

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.preprocess(x)
        
        x = torch.FloatTensor(x)

        out = self.model(x)
        return out.detach().numpy().astype('float')
