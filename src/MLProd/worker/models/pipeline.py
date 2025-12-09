from .model import Model

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

import numpy as np
import logging
import torch
import json
import joblib
import os

FILE_METADATA: str = "metadata.json"
FILE_MMS: str = "mms.model"
FILE_SKB: str = "skb.model"
FILE_MODEL: str = "neuralnet.model"

DEFAULT_MODELS_PATH = Path("./models/")


class PipelineModel:
    """This is a pipeline model.

    It allows to aggregate all the models in a single call where the pre-processing
    and the models are applied in order. It also manage the loading of models from
    disk.
    """

    def __init__(self, path: Path = DEFAULT_MODELS_PATH) -> None:
        """Creates a new model by loading the required data from the given path.

        :param path:
            The path must exists, be a folder and containing four files:
            - metadata.json
            - mms.model
            - skb.model
            - neuralnet.model
            These files are produced both by the notebook and by the training tasks.
        """
        super().__init__()

        self.path: Path = path

        logging.info(f"loading model from path {self.path}")

        # path setup
        self.path_metadata: str = str(os.path.join(self.path, FILE_METADATA))
        self.path_mms: str = str(os.path.join(self.path, FILE_MMS))
        self.path_skb: str = str(os.path.join(self.path, FILE_SKB))
        self.path_model: str = str(os.path.join(self.path, FILE_MODEL))

        logging.info(f"Load metadata from {self.path_metadata}")

        # files loading
        with open(self.path_metadata, "r") as f:
            self.metadata: dict = json.load(f)

        # pre-processing model creation
        logging.info(f"Loading pre-process models from {self.path_mms} {self.path_skb}")

        self.mms: MinMaxScaler = joblib.load(self.path_mms)
        self.skb: SelectKBest = joblib.load(self.path_skb)

        logging.info(f"Loading model from {self.path_model}")

        nn_state_dict = torch.load(self.path_model)
        self.model: Model = Model(self.metadata["x_output"])
        self.model.load_state_dict(nn_state_dict)

        logging.info("All artifacts loaded")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Applies the pipeline to the input data.

        :param x:
            Input values. Can be a single record or multiple records.

        :return:
            A score value for each input record.
        """
        x_temp = self.mms.transform(x)
        x_temp = self.skb.transform(x_temp)

        x_temp = torch.FloatTensor(x_temp)

        y = self.model(x_temp)
        return y.detach().numpy().astype("float")
