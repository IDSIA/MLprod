import numpy as np

import math

from api.requests import LocationData
from datas.utils import sample_bool, sample_float


def generate_location_data(
    r: np.random.Generator, 
    coordinates: tuple[float, float]= (0.0, 0.0),
    location_distance_min: float=1, # this is in KM!
    location_distance_max: float=1, # this is in KM!
    threshold_child: float=0.5,
    threshold_breakfast: float=0.5,
    threshold_lunch: float=0.5,
    threshold_dinner: float=0.5,
    price_min: float=50,
    price_max: float=500,
    threshold_pool: float=0.5,
    threshold_spa: float=0.5,
    threshold_animals: float=0.5,
    threshold_lake: float=0.5,
    threshold_mountain: float=0.5,
    threshold_sport: float=0.5,
    family_min: float=0.0,
    family_max: float=1.0,
    outdoor_min: float=0.0,
    outdoor_max: float=1.0,
    food_min: float=0.0,
    food_max: float=1.0,
    leisure_min: float=0.0,
    leisure_max: float=1.0,
    service_min: float=0.0,
    service_max: float=1.0,
    score_min: float=0.0,
    score_max: float=1.0,
    a: float=1.0,
    b: float=1.0,
) -> LocationData:

    """This function will generate the location in a synthtetic way. The objective is
    create possible numeric descriptors for each location.

    Act on paramters ``a`` and ``b`` for distribution skew.
    """
    # coordinates are expressed in km...
    lat, lon = coordinates

    # we can use polar-coordinate-conversion to compute a point in the given range
    angle = sample_float(r, 0, 2 * math.pi, a, b)
    kms = sample_float(r, location_distance_min, location_distance_max, a, b) 
    lat += kms * math.cos(angle)
    lon += kms * math.sin(angle)

    children = sample_bool(r, threshold_child, a, b)
    breakfast = sample_bool(r, threshold_breakfast, a, b)
    lunch = sample_bool(r, threshold_lunch, a, b)
    dinner = sample_bool(r, threshold_dinner, a, b)

    price = sample_float(r, price_min, price_max, a, b)

    pool = sample_bool(r, threshold_pool, a, b)
    spa = sample_bool(r, threshold_spa, a, b)
    animals = sample_bool(r, threshold_animals, a, b)
    lake = sample_bool(r, threshold_lake, a, b)
    mountain = sample_bool(r, threshold_mountain, a, b)
    sport = sample_bool(r, threshold_sport, a, b)

    family_rating = sample_float(r, family_min, family_max, a, b)
    outdoor_rating = sample_float(r, outdoor_min, outdoor_max, a, b)
    food_rating = sample_float(r, food_min, food_max, a, b)
    leisure_rating = sample_float(r, leisure_min, leisure_max, a, b)
    service_rating = sample_float(r, service_min, service_max, a, b)
    user_score = sample_float(r, score_min, score_max, a, b)

    return LocationData(
        lat=lat,
        lon=lon,
        children=children,
        breakfast=breakfast,
        lunch=lunch,
        dinner=dinner,
        price=price,
        pool=pool,
        spa=spa,
        animals=animals,
        lake=lake,
        mountain=mountain,
        sport=sport,
        family_rating=family_rating,
        outdoor_rating=outdoor_rating,
        food_rating=food_rating,
        leisure_rating=leisure_rating,
        service_rating=service_rating,
        user_score=user_score,
    )
