import numpy as np
import requests

from time import sleep
from dotenv import load_dotenv

import logging
import os

logging.basicConfig(level=logging.INFO)

load_dotenv()

DOMAIN = os.getenv('DOMAIN')

if __name__ == '__main__':

    r = np.random.default_rng()

    while True:
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
        logging.info(f'Sleeping for {T1}')

        sleep(T1)
        url_get = f'http://mlpapi.{DOMAIN}:4789/result/{tid}'

        ret = requests.get(url_get)
        
        y = ret.json()['y']

        logging.info(f'Task id={tid} Y={y}')
        logging.info(f'Sleeping for {T2}')
        sleep(T2)
