from pydantic import BaseModel
from datetime import date


class PredictionInput(BaseModel):
    """Class that defines the input data for an inference."""
    x: float


class PredictionOutput(BaseModel):
    """Class that defines the output data of an inference.
    
    Field ``y`` could be null if the inference is still working."""
    task_id: str
    status : str
    y: float | None

class RequestData(BaseModel):
    people_num: int
    people_age: list[int] # for each people
    children: int
    budget: float
    dst_latitude: float
    dst_longitue: float
    dst_range: float # km
    time_arrival: date
    nights: int
    spa: bool
    pool: bool
    pet_friendly: bool
    lakes: bool
    mountains: bool
    sport: bool
