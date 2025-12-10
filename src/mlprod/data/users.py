import numpy as np
import pandas as pd

from mlprod.api.requests import UserData
from mlprod.data.configs import UserConfig
from mlprod.data.utils import sample_bool, sample_float, sample_int

from pathlib import Path
from datetime import date


def read_user_config(config: Path) -> list[UserConfig]:
    """Read a config file in TSV format to generate different kind of users.

    Returns a list of settings that can be used with the `generate_user_data_from_conf` function.

    :param config:
        A valid path to a TSV file.
    """
    user_conf = pd.read_csv(config, sep="\t")
    user_settings = [UserConfig(**row.to_dict()) for _, row in user_conf.iterrows()]
    return user_settings


def generate_user_data(
    r: np.random.Generator,
    config: UserConfig,
    start_date: np.datetime64,
) -> UserData:
    """This function will generate the input user data  in a synthetic way.

    The objective is to simulate possible requests from the users on the hypothetical web interface
    and start the inferences on the possible.

    Act on parameters ``a`` and ``b`` for distribution skew.
    """
    if config.people_min < config.people_max:
        people_num = sample_int(r, config.people_min, config.people_max)[0]
    else:
        people_num = config.people_min

    ages = sample_int(r, config.age_min, config.age_max, people_num)

    if config.minor_age > 0:
        children_num = any(ages < config.minor_age)
    else:
        children_num = 0

    budget = sample_float(r, config.budget_min, config.budget_max)[0]

    # lat, lon = sample_list(r, LOCATIONS, a, b)
    # ran = sample_float(r, range_min, range_max, a, b)

    nights = sample_int(r, config.nights_min, config.nights_max)[0]
    time_arrival: np.datetime64 = (
        start_date
        + sample_int(
            r,
            config.start_date_tolerance_min,
            config.start_date_tolerance_max,
        )[0]
    )

    spa = sample_bool(r, config.spa_thr)
    pool = sample_bool(r, config.pool_thr)
    pet_friendly = sample_bool(r, config.pet_friendly_thr)
    lake = sample_bool(r, config.lakes_thr)
    mountain = sample_bool(r, config.mountains_thr)
    sport = sample_bool(r, config.sport_thr)

    return UserData(
        name=config.meta_comment,
        people_num=people_num,
        people_age=ages.tolist(),
        children_num=children_num,
        budget=budget,
        time_arrival=time_arrival.astype(date),
        nights=nights,
        spa=spa,
        pool=pool,
        pet_friendly=pet_friendly,
        lake=lake,
        mountain=mountain,
        sport=sport,
    )
