from pydantic import BaseModel


class UserConfig(BaseModel):
    """Parameters defining a type of user."""

    meta_comment: str
    meta_n: int

    people_min: int = 1
    people_max: int = 10
    age_min: int = 0
    age_max: int = 99
    minor_age: int = 18
    budget_min: float = 50
    budget_max: float = 10000
    nights_min: int = 1
    nights_max: int = 14
    spa_thr: float = 0.5
    pool_thr: float = 0.5
    pet_friendly_thr: float = 0.5
    lakes_thr: float = 0.5
    mountains_thr: float = 0.5
    sport_thr: float = 0.5

    budget_tolerance: float = 0.0
    facilities_tolerance: float = 0.0
    environment_tolerance: float = 0.0
    family_tolerance: float = 1.0
    weight_spa: float = 1.0
    weight_pool: float = 1.0
    weight_pet: float = 1.0
    weight_lake: float = 1.0
    weight_mountains: float = 1.0
    weight_sport: float = 1.0
    weight_score_facilities: float = 1.0
    weight_score_environment: float = 1.0
    weight_score_users: float = 1.0
    exploration_tolerance: float = 0.5
    ignore_tolerance: float = 0.99

    start_date_tolerance_min: int = 1
    start_date_tolerance_max: int = 30


class LocationConfig(BaseModel):
    """Parameters defining a location."""

    meta_comment: str
    meta_n: int
    threshold_child: float = 0.5
    threshold_breakfast: float = 0.5
    threshold_lunch: float = 0.5
    threshold_dinner: float = 0.5
    price_min: float = 50
    price_max: float = 500
    threshold_pool: float = 0.5
    threshold_spa: float = 0.5
    threshold_animals: float = 0.5
    threshold_lake: float = 0.5
    threshold_mountain: float = 0.5
    threshold_sport: float = 0.5
    family_min: float = 0.0
    family_max: float = 1.0
    outdoor_min: float = 0.0
    outdoor_max: float = 1.0
    food_min: float = 0.0
    food_max: float = 1.0
    leisure_min: float = 0.0
    leisure_max: float = 1.0
    service_min: float = 0.0
    service_max: float = 1.0
    score_min: float = 0.0
    score_max: float = 1.0
