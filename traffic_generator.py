from time import sleep

from api.requests import LocationData, UserData
from datas import read_user_config, generate_user_data_from_config, UserLabeller

import numpy as np
import requests
import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
)


def setup_arguments():
    """Defines the available input parameters"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Set fixed seed', default=42, type=int)
    parser.add_argument('-n', help='Number of requests to do, 0 means infinite', default=None, type=int)
    parser.add_argument('--no-sleep', help='If set, no sleeps are used', default=False, action='store_true')
    parser.add_argument('--config', help='User config file to use', default='./config/user.tsv', type=str)
    parser.add_argument('-p', help='Number of parallel generators.', default=1, type=int)

    return parser.parse_args()


def sleepy(thread: int, time: int, flag: bool=True):
    """Add some delay in the system.
    
    :param time:
      The amount of seconds to wait
    
    :param flag:
      If True (default value), the sleepy is executed, otherwise no 
    """
    if flag:
        logging.info(f'{thread:02} Sleeping for {time*1000:.0f}ms')
        sleep(time)


def inference_start(url: str, user: UserData) -> str:
    data = user.dict()
    data['time_arrival'] = str(data['time_arrival'])
    p_data = requests.post(
        url=f'{url}/inference/start', 
        headers={
            'accept': 'application/json',
            'Content-Type': 'application/json',
        },
        json=data,
    )

    return p_data.json()['task_id']


def inference_status(url: str, task_id: str) -> bool:
    g_status = requests.get(
        url=f'{url}/inference/status/{task_id}',
        headers={
            'accept': 'application/json',
        },
    )

    status = g_status.json()
    return status['status']


def inference_results(url: str, task_id: str) -> dict:
    g_results = requests.get(
        url=f'{url}/inference/results/{task_id}',
        headers={
            'accept': 'application/json',
        },
    )

    data = g_results.json()
    return [LocationData(**d) for d in data]


def make_choice(url: str, task_id: str, location_id: int) -> dict:
    u_result = requests.put(
        url=f'{url}/inference/select/',
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


def exec(args) -> None:
    """"""
    N, seed, config, url, thread = args

    r = np.random.default_rng(seed=seed)
    user_configs = read_user_config(config)
    ul = UserLabeller()

    n = 1 if N is None else N
    while n > 0:
        T1 = r.integers(2, 30) * .1
        T2 = r.integers(2, 30) * .1
        T3 = r.integers(2, 30) * .1

        # choose next user ----------------------------------------
        user_config = r.choice(user_configs)

        logging.info(f'{thread:02} User: {user_config["name"]}')

        sleepy(thread, T1)

        user = generate_user_data_from_config(r, user_config)

        # send new inference request ------------------------------
        task_id = inference_start(url, user)

        logging.info(f'{thread:02} Task id assigned: {task_id}')

        # send task get request -----------------------------------
        done = False
        while not done:
            sleepy(thread, T2)
            status = inference_status(url, task_id)

            logging.info(f'{thread:02} Request status: {status}')
            done = status == 'SUCCESS'


        locations = inference_results(url, task_id)

        # Make choice ---------------------------------------------
        labels = ul(r, user, locations)
        location_ids = np.array([l.location_id for l in locations])

        location_id = int(r.choice(location_ids[labels == 1]))
        
        logging.info(f'{thread:02} Choosen location with id {location_id}')

        # Register choice -----------------------------------------
        sleepy(thread, T3)
        make_choice(url, task_id, location_id)

        # next request --------------------------------------------
        if N is not None:
            n -= 1


if __name__ == '__main__':
    from dotenv import load_dotenv
    from multiprocessing import Pool, set_start_method
    set_start_method('spawn')

    # Environment variables are controlled through a .env file
    load_dotenv()

    if 'URL' in os.environ:
        URL = os.environ.get('URL')
    else:
        DOMAIN = os.environ.get('DOMAIN', 'localhost')
        URL = f'http://mlpapi.{DOMAIN}'

    args = setup_arguments()

    r = np.random.default_rng(args.seed)

    workers = min(os.cpu_count(), args.p)

    params = [(args.n, args.seed+i, args.config, URL, i+1) for i in range(workers)]

    logging.info(f'Starting traffic generation with {workers} worker(s)')

    with Pool(processes=workers) as pool:
        pool.map(exec, params)
