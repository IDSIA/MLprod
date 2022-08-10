__all__ = [
    'Model',
    'train_model',
    'evaluate',
]

from worker.models.wrapper import InferenceModel as Model
from worker.models.train import train_model, evaluate
