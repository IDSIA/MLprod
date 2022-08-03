from pydantic import BaseModel
from datetime import date


class InferenceStatus(BaseModel):
    """Class that defines the output data of an inference."""
    task_id: str
    status : str


class UserData(BaseModel):
    people_num: int
    people_age: list[int] # for each people
    children: int
    budget: float
    nights: int
    time_arrival: date
    pool: bool
    spa: bool
    pet_friendly: bool
    lake: bool
    mountain: bool
    sport: bool


class LocationData(BaseModel):
    location_id: int = 0
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


class LabelData(BaseModel):
    task_id: str
    location_id: int

class ContentInfo(BaseModel):
    locations: int
    users: int
