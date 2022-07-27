from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import joblib


class PreProcess:

    def __init__(self, mms_path: str, skb_path: str) -> None:

        self.mms: MinMaxScaler = joblib.load(mms_path)
        self.skb: SelectKBest = joblib.load(skb_path)

    def __call__(self, instance: np.ndarray) -> np.ndarray:
        # TODO: add ages features !

        instance = self.mms.transform(instance)
        instance = self.skb.transform(instance)

        return instance
