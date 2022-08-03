from time import sleep
from dotenv import load_dotenv
from api.requests import LocationData, UserData

from datas import read_user_config, generate_user_data_from_config, UserLabeller

import numpy as np
import requests
import argparse
import logging
import os

# Environment variables are controlled through a .env file
load_dotenv()


if 'URL' in os.environ:
    URL = os.environ.get('URL')
else:
    DOMAIN = os.environ.get('DOMAIN', 'localhost')
    URL = f'http://mlpapi.{DOMAIN}'


def setup_arguments():
    """Defines the available input parameters"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Set fixed seed', default=None, type=int)
    parser.add_argument('-n', help='Number of requests to do, 0 means infinite', default=None, type=int)
    parser.add_argument('--no-sleep', help='If set, no sleeps are used', default=False, action='store_true')
    parser.add_argument('--config', help='User config file to use', default='./config/user.tsv', type=str)

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


def inference_start(user: UserData) -> str:
    data = user.dict()
    data['time_arrival'] = str(data['time_arrival'])
    p_data = requests.post(
        url=f'{URL}/inference/start', 
        headers={
            'accept': 'application/json',
            'Content-Type': 'application/json',
        },
        json=data,
    )

    return p_data.json()['task_id']


def inference_status(task_id: str) -> bool:
    g_status = requests.get(
        url=f'{URL}/inference/status/{task_id}',
        headers={
            'accept': 'application/json',
        },
    )

    status = g_status.json()
    logging.info(f'Request status: {status["status"]}')
    return status['status'] == 'SUCCESS'


def inference_results(task_id: str) -> dict:
    g_results = requests.get(
        url=f'{URL}/inference/results/{task_id}',
        headers={
            'accept': 'application/json',
        },
    )

    data = g_results.json()
    return [LocationData(**d) for d in data]


def make_choice(task_id: str, location_id: int) -> dict:
    u_result = requests.put(
        url=f'{URL}/inference/select/',
        headers={
            'accept': 'application/json',
            'Content-Type': 'application/json',
        },
        json={
            'task_id': task_id,
            'location_id': location_id,
        },
    )

    return u_result.json()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    args = setup_arguments()

    user_configs = read_user_config(args.config)
    ul = UserLabeller()

    r = np.random.default_rng(args.seed)

    n = 1 if args.n is None else args.n

    while n > 0:
        T1 = r.integers(2, 30) * .1
        T2 = r.integers(2, 30) * .1
        T3 = r.integers(2, 30) * .1

        # choose next user ----------------------------------------
        user_config = r.choice(user_configs)

        logging.info(f'User: {user_config["name"]}')

        sleepy(T1)

        user = generate_user_data_from_config(r, user_config)

        # send new inference request ------------------------------
        task_id = inference_start(user)

        logging.info(f'Task id assigned: {task_id}')

        # send task get request -----------------------------------
        done = False
        while not done:
            sleepy(T2)
            done = inference_status(task_id)

        locations = inference_results(task_id)

        # Make choice ---------------------------------------------
        labels = ul(r, user, locations)
        location_ids = np.array([l.location_id for l in locations])

        location_id = int(r.choice(location_ids[labels == 1]))
        
        logging.info(f'Choosen location with id {location_id}')

        # Register choice -----------------------------------------
        sleepy(T3)
        make_choice(task_id, location_id)

        # next request --------------------------------------------
        if args.n is not None:
            n -= 1
