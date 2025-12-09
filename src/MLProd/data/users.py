import numpy as np
import pandas as pd

from MLProd.api.requests import UserData
from MLProd.data.labels import UserLabeller
from MLProd.data.utils import sample_bool, sample_float, sample_int

from pathlib import Path


def read_user_config(config: Path) -> list[UserData]:
    """Read a config file in TSV format to generate different kind of users.

    Returns a list of settings that can be used with the `generate_user_data_from_conf` function.

    :param config:
        A valid path to a TSV file.
    """
    user_conf = pd.read_csv(config, sep="\t")
    user_settings = []
    for _, row in user_conf.iterrows():
        user_dict = {
            "name": row["meta_comment"],
            "qnt": row["meta_dist"],
            "settings": row.iloc[2:16].to_dict(),
            "labels": row.iloc[17:].to_dict(),
        }
        user_settings.append(user_dict)

    return [UserData(**u) for u in user_settings]


def generate_user_labeller_from_config(conf: dict) -> UserLabeller:
    """Utility wrapper for `UserLabeller()` constructor method."""
    return UserLabeller(**conf["labels"])


def generate_user_data_from_config(r: np.random.Generator, conf: dict) -> UserData:
    """Utility wrapper for `generate_user_data()` function."""
    return generate_user_data(r=r, **conf["settings"])


def generate_user_data(
    r: np.random.Generator,
    people_min: int = 1,
    people_max: int = 10,
    age_min: int = 0,
    age_max: int = 99,
    minor_age: int = 18,
    budget_min: float = 50,
    budget_max: float = 10000,
    nights_min: int = 1,
    nights_max: int = 14,
    spa_thr: float = 0.5,
    pool_thr: float = 0.5,
    pet_friendly_thr: float = 0.5,
    lakes_thr: float = 0.5,
    mountains_thr: float = 0.5,
    sport_thr: float = 0.5,
    start_date: np.datetime64 = np.datetime64("2022-06-01"),
    start_date_tolerance_min: int = 1,
    start_date_tolerance_max: int = 30,
) -> UserData:
    """This function will generate the input user data  in a synthetic way.

    The objective is to simulate possible requests from the users on the hypothetical web interface
    and start the inferences on the possible.

    Act on parameters ``a`` and ``b`` for distribution skew.
    """
    if people_min < people_max:
        people_num = sample_int(r, people_min, people_max)[0]
    else:
        people_num = people_min

    ages = sample_int(r, age_min, age_max, people_num)

    if minor_age > 0:
        children_num = any(ages < minor_age)
    else:
        children_num = 0

    budget = sample_float(r, budget_min, budget_max)[0]

    # lat, lon = sample_list(r, LOCATIONS, a, b)
    # ran = sample_float(r, range_min, range_max, a, b)

    nights = sample_int(r, nights_min, nights_max)[0]
    time_arr = (
        np.datetime64(start_date)
        + sample_int(r, start_date_tolerance_min, start_date_tolerance_max)[0]
    )

    spa = sample_bool(r, spa_thr)
    pool = sample_bool(r, pool_thr)
    pet_friendly = sample_bool(r, pet_friendly_thr)
    lake = sample_bool(r, lakes_thr)
    mountain = sample_bool(r, mountains_thr)
    sport = sample_bool(r, sport_thr)

    return UserData(
        name="",
        people_num=people_num,
        people_age=ages.tolist(),
        children_num=children_num,
        budget=budget,
        time_arrival=time_arr,
        nights=nights,
        spa=spa,
        pool=pool,
        pet_friendly=pet_friendly,
        lake=lake,
        mountain=mountain,
        sport=sport,
    )
