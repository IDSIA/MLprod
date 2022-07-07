from pydantic import BaseModel


class PredictionInput(BaseModel):
    x: float


class PredictionOutput(BaseModel):
    task_id: str
    status : str
    y: float | None
