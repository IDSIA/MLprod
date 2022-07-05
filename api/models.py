from pydantic import BaseModel


class PredictionInput(BaseModel):
    x: float


class Inputs(BaseModel):
    x: float
    y: float


class TaskOutput(BaseModel):
    task_id: str
    status : str


class TaskResult(TaskOutput):
    y: float
