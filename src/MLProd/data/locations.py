import numpy as np
import pandas as pd

from MLProd.api.requests import LocationData
from MLProd.data.configs import LocationConfig
from MLProd.data.utils import sample_bool, sample_float

from pathlib import Path


def read_location_config(config: Path) -> list[LocationConfig]:
    """Read a configuration from a given path.

    This function will read a file in TSV format that contains all the information to
    generate different kind of location.
    Return a list of settings that can be used with the `generate_location_data_from_config`
    function.

    :param config:
        A valid path to a TSV file.
    """
    loc_conf = pd.read_csv(config, sep="\t")
    loc_settings = [LocationConfig(**row.to_dict()) for _, row in loc_conf.iterrows()]
    return loc_settings


def generate_location_data_from_config(
    r: np.random.Generator, conf: LocationConfig
) -> LocationData:
    """Utility wrapper for `generate_location_data()` function."""
    return generate_location_data(r, conf)


def generate_location_data(
    r: np.random.Generator,
    conf: LocationConfig,
) -> LocationData:
    """Generates the location in a synthetic way.

    The objective is create possible numeric descriptors for each location.

    Act on parameters ``a`` and ``b`` for distribution skew.
    """
    children = sample_bool(r, conf.threshold_child)
    breakfast = sample_bool(r, conf.threshold_breakfast)
    lunch = sample_bool(r, conf.threshold_lunch)
    dinner = sample_bool(r, conf.threshold_dinner)

    price = sample_float(r, conf.price_min, conf.price_max)[0]

    pool = sample_bool(r, conf.threshold_pool)
    spa = sample_bool(r, conf.threshold_spa)
    animals = sample_bool(r, conf.threshold_animals)
    lake = sample_bool(r, conf.threshold_lake)
    mountain = sample_bool(r, conf.threshold_mountain)
    sport = sample_bool(r, conf.threshold_sport)

    family_rating = sample_float(r, conf.family_min, conf.family_max)[0]
    outdoor_rating = sample_float(r, conf.outdoor_min, conf.outdoor_max)[0]
    food_rating = sample_float(r, conf.food_min, conf.food_max)[0]
    leisure_rating = sample_float(r, conf.leisure_min, conf.leisure_max)[0]
    service_rating = sample_float(r, conf.service_min, conf.service_max)[0]
    user_score = sample_float(r, conf.score_min, conf.score_max)[0]

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
