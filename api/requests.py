from pydantic import BaseModel


class PredictionInput(BaseModel):
    """Class that defines the input data for an inference."""
    x: float


class PredictionOutput(BaseModel):
    """Class that defines the output data of an inference.
    
    Field ``y`` could be null if the inference is still working."""
    task_id: str
    status : str
    y: float | None
