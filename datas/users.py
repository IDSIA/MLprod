import numpy as np
from api.requests import UserData

from datas.coordinates import LOCATIONS
from datas.utils import sample_bool, sample_float, sample_int, sample_list


def generate_user_data(
    r: np.random.Generator, 
    people_min: int=1,
    people_max: int=10,
    age_min: int=0,
    age_max: int=99,
    minor_age: int=18,
    budget_min: float=1000,
    budget_max: float=10000,
    range_min: float=0.0,
    range_max: float=100.0,
    nights_min: int=1,
    nights_max: int=14,
    start_date: np.datetime64='2022-06-01',
    start_date_tolerance_min: int=0,
    start_date_tolerance_max: int=30,
    spa_thr: float=.5,
    pool_thr: float=.5,
    pet_friendly_thr: float=.5,
    lakes_thr: float=.5,
    mountains_thr: float=.5,
    sport_thr: float=.5,
    a: float=1.0,
    b: float=1.0,
) -> UserData:
    """This function will generate the input user data  in a synthtetic way. The objective
    is to simulate possible requests from the users on the hypotetical web interface
    and start the inferences on the possible.

    Act on paramters ``a`` and ``b`` for distribution skew.
    """
    people = sample_int(r, people_min, people_max, a, b)
    ages = sample_int(r, age_min, age_max, a, b, people)

    children = any(ages < minor_age)

    budget = sample_float(r, budget_min, budget_max, a, b)

    lat, lon = sample_list(r, LOCATIONS, a, b)
    
    ran = sample_float(r, range_min, range_max, a, b)
    nights = sample_int(r, nights_min, nights_max, a, b)
    time_arr = str(np.datetime64(start_date) + sample_int(r, start_date_tolerance_min, start_date_tolerance_max, a,b))
    
    spa = sample_bool(r,spa_thr, a, b)
    pool = sample_bool(r,pool_thr, a, b)
    pet_friendly = sample_bool(r, pet_friendly_thr, a, b)
    lakes = sample_bool(r,lakes_thr, a, b)
    mountains = sample_bool(r, mountains_thr, a, b)
    sport = sample_bool(r,sport_thr, a, b)

    return UserData(
        people_num=people,
        people_age=ages.tolist(),
        children=children,
        budget=budget,
        dest_lat=lat,
        dest_lon=lon,
        dst_range=ran,
        time_arrival=time_arr,
        nights=nights,
        spa=spa,
        pool=pool,
        pet_friendly=pet_friendly,
        lakes=lakes,
        mountains=mountains,
        sport=sport,
    )
