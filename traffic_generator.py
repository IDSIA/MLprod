import numpy as np
import requests

from time import sleep
from dotenv import load_dotenv

from api.requests import RequestData
from datas.locations import LOCATIONS

import argparse
import logging
import os

# Environment variables are controlled through a .env file
load_dotenv()

DOMAIN = os.getenv('DOMAIN', 'localhost')

def setup_arguments():
    """Defines the available input parameters"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Set fixed seed', default=None, type=int)
    parser.add_argument('-n', help='Number of requests to do, 0 means infinite', default=None, type=int)
    parser.add_argument('--no-sleep', help='If set, no sleeps are used', default=False, action='store_true')

    return parser.parse_args()


def sleepy(time:int, flag:bool=True):
    """Add some delay in the system.
    
    :param time:
      The amount of seconds to wait
    
    :param flag:
      If True (default value), the sleepy is executed, otherwise no 
    """
    if flag:
        logging.info(f'Sleeping for {time}')
        sleep(time)


def generate_request_data(
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
    spa_thr: float=.5,
    pool_thr: float=.5,
    pet_friendly_thr: float=.5,
    lakes_thr: float=.5,
    mountains_thr: float=.5,
    sport_thr: float=.5,
) -> RequestData:
    """This function will generate the input data in a synthtetic way. The objective
    is to simulate possible requests from the users on the hypotetical web interface
    and start the inferences on the possible 
    """
    # TODO: use different generator to have different skewed distributions
    people = r.integers(people_min, people_max, type='int'),
    ages = r.integers(age_min, age_max, people, type='int').tolist()
    children = any(ages < minor_age)

    budget = r.uniform(budget_min, budget_max, type='float')

    lat, lon = r.choice(LOCATIONS)
    ran = r.uniform(range_min, range_max)
    nights = r.integers(nights_min, nights_max, type='int')
    
    spa = r.uniform() > spa_thr
    pool = r.uniform() > pool_thr
    pet_friendly = r.uniform() > pet_friendly_thr
    lakes = r.uniform() > lakes_thr
    mountains = r.uniform() > mountains_thr
    sport = r.uniform() > sport_thr

    return RequestData(
        people_num=people,
        people_age=ages,
        children=children,
        budget=budget,
        dst_latitude=lat,
        dst_longitue=lon,
        dst_range=range,
        time_arrival=ran,
        nights=nights,
        spa=spa,
        pool=pool,
        pet_friendly=pet_friendly,
        lakes=lakes,
        mountains=mountains,
        sport=sport,
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    args = setup_arguments()

    r = np.random.default_rng(args.seed)

    n = 1 if args.n is None else args.n
    while n > 0:
        X = r.uniform(-10, 10)

        T1 = r.integers(1, 3)
        T2 = r.integers(0, 5)

        # send new inference request ------------------------------
        url_post = f'http://mlpapi.{DOMAIN}:4789/pred'
        data = generate_request_data(r)

        logging.info(f'Request X={X}')

        ret = requests.post(url_post, json=data.dict())

        logging.info(f'Return code: {ret.status_code}')

        tid = ret.json()['task_id']
        tst = ret.json()['status']

        # ---------------------------------------------------------
        
        # send task get request -----------------------------------
        logging.info(f'Task id={tid} task status={tst}')
        sleepy(T1, not args.no_sleep)

        url_get = f'http://mlpapi.{DOMAIN}:4789/result/{tid}'

        ret = requests.get(url_get)
        y = ret.json()['y']

        logging.info(f'Task id={tid} Y={y}')
        sleepy(T2, not args.no_sleep)

        # ---------------------------------------------------------

        # next request --------------------------------------------
        if args.n is not None:
            n -= 1
