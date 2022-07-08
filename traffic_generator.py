import numpy as np
import requests

from time import sleep
from dotenv import load_dotenv

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

        url_post = f'http://mlpapi.{DOMAIN}:4789/pred'
        data = {
            'x': X
        }

        logging.info(f'Request X={X}')

        ret = requests.post(url_post, json=data)

        logging.info(f'Return code: {ret.status_code}')

        tid = ret.json()['task_id']
        tst = ret.json()['status']

        logging.info(f'Task id={tid} task status={tst}')
        sleepy(T1, not args.no_sleep)

        url_get = f'http://mlpapi.{DOMAIN}:4789/result/{tid}'

        ret = requests.get(url_get)
        y = ret.json()['y']

        logging.info(f'Task id={tid} Y={y}')
        sleepy(T2, not args.no_sleep)

        if args.n is not None:
            n -= 1
