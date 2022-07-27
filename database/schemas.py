from pydantic import BaseModel
from datetime import datetime


class PredictionBase(BaseModel):
    task_id: str
    x: float = None
    status: str = ''


class PredictionCreate(PredictionBase):
    pass


class Prediction(PredictionBase):
    y: float | None = None

    time_post: datetime | None
    time_get: datetime | None

    class Config:
        orm_mode = True
