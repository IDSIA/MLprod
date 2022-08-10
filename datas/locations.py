import numpy as np
import pandas as pd

from api.requests import LocationData
from datas.utils import sample_bool, sample_float


def read_location_config(config: str) -> list[dict]:
    """This function will read a file in TSV format that contains all the information
    to generate different kind of location. Return a list of settings that can be used 
    with the `generate_location_data_from_config` function.
    
    :param config:
        A valid path to a TSV file.
    """
    loc_conf = pd.read_csv(config, sep='\t')
    loc_settings = []
    for _, row in loc_conf.iterrows():
        loc_dict = {
            'name': row['meta_comment'],
            'qnt': row['meta_n'],
            'settings': row.iloc[2:].to_dict()
        }
        loc_settings.append(loc_dict)

    return loc_settings


def generate_location_data_from_config(r: np.random.Generator, conf: dict) -> LocationData:
    """Utility wrapper for `generate_location_data()` function."""
    return generate_location_data(r=r, **conf['settings'])


def generate_location_data(
    r: np.random.Generator, 
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
) -> LocationData:
    """This function will generate the location in a synthtetic way. The objective is
    create possible numeric descriptors for each location.

    Act on paramters ``a`` and ``b`` for distribution skew.
    """

    children = sample_bool(r, threshold_child)
    breakfast = sample_bool(r, threshold_breakfast)
    lunch = sample_bool(r, threshold_lunch)
    dinner = sample_bool(r, threshold_dinner)

    price = sample_float(r, price_min, price_max)

    pool = sample_bool(r, threshold_pool)
    spa = sample_bool(r, threshold_spa)
    animals = sample_bool(r, threshold_animals)
    lake = sample_bool(r, threshold_lake)
    mountain = sample_bool(r, threshold_mountain)
    sport = sample_bool(r, threshold_sport)

    family_rating = sample_float(r, family_min, family_max)
    outdoor_rating = sample_float(r, outdoor_min, outdoor_max)
    food_rating = sample_float(r, food_min, food_max)
    leisure_rating = sample_float(r, leisure_min, leisure_max)
    service_rating = sample_float(r, service_min, service_max)
    user_score = sample_float(r, score_min, score_max)

    return LocationData(
        children=children,
        breakfast=breakfast,
        lunch=lunch,
        dinner=dinner,
        price=price,
        has_pool=pool,
        has_spa=spa,
        animals=animals,
        near_lake=lake,
        near_mountains=mountain,
        has_sport=sport,
        family_rating=family_rating,
        outdoor_rating=outdoor_rating,
        food_rating=food_rating,
        leisure_rating=leisure_rating,
        service_rating=service_rating,
        user_score=user_score,
    )
