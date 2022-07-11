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


class UserData(BaseModel):
    people_num: int
    people_age: list[int] # for each people
    children: int
    budget: float
    dest_lat: float
    dest_lon: float
    dst_range: float # km
    time_arrival: date
    nights: int
    spa: bool
    pool: bool
    pet_friendly: bool
    lakes: bool
    mountains: bool
    sport: bool


class LocationData(BaseModel):
    lat: float
    lon: float
    children: bool
    breakfast: bool
    lunch: bool
    dinner: bool
    price: float
    pool: bool = False
    spa: bool = False
    animals: bool = False
    lake: bool = False
    mountain: bool = False
    sport: bool = False
    family_rating: float
    outdoor_rating: float
    food_rating: float
    leisure_rating: float
    service_rating: float
    user_score: float
