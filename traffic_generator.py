import numpy as np
import requests

from time import sleep
from dotenv import load_dotenv

import argparse
import logging
import os

load_dotenv()

DOMAIN = os.getenv('DOMAIN')

def setup_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Use fixed seed', default=None, type=int)
    parser.add_argument('-n', help='Number of requests to do', default=0, type=int)
    parser.add_argument('--no-sleep', help='If set, no sleeps are used', default=False, action='store_true')

    return parser.parse_args()


def sleepy(time, flag):
    if flag:
        logging.info(f'Sleeping for {time}')
        sleep(time)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )

    args = setup_arguments()

    SEED = args.seed
    N = args.n

    if SEED is None:
        r = np.random.default_rng()
    else:
        r = np.random.default_rng(SEED)

    n = N
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

        if N > 0:
            n -= 1
